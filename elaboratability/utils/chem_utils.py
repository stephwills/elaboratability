import os

import numpy as np
from rdkit import RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, DataStructs
from rdkit.ML.Cluster import Butina
from tqdm import tqdm


def distance_matrix(mols):

    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) for mol in mols]
    nfps = len(fps)

    dists = []
    # calculate the distance matrix
    for i in tqdm(range(nfps), total=nfps, position=0, leave=True):
        for j in range(i):
            tanimoto = DataStructs.FingerprintSimilarity(
                fps[i], fps[j], metric=DataStructs.TanimotoSimilarity
            )
            dists.append(1 - tanimoto)

    return dists

def butina_cluster_picks(mols, threshold):
    nfps = len(mols)

    # calculate the distance matrix
    dists = distance_matrix(mols)
    print("Distance matrix calculated")

    # cluster the data
    cs = list(Butina.ClusterData(dists, nfps, threshold, isDistData=True))
    print(f"{len(cs)} clusters found")

    # the first mol of each cluster is the centroid - retrieve smiles
    picks = []
    for cluster in cs:
        centroid = cluster[0]
        picks.append(centroid)

    return picks


def get_all_coordinates(mols, mol_ids, atomic_num):
    all_coordinates = np.array([[0,0,0]])
    confIds = []
    atomIds = []
    molIds = []
    for confid, (mol, mol_id) in tqdm(enumerate(zip(mols, mol_ids)), total=len(mols), position=0, leave=True):
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == atomic_num:
                idx = atom.GetIdx()
                coords = np.array(conf.GetAtomPosition(idx))
                all_coordinates = np.vstack((all_coordinates, coords))
                atomIds.append(idx)
                confIds.append(confid)
                molIds.append(mol_id)

    all_coordinates = np.delete(all_coordinates, (0), axis=0)
    print(f'Coords shape for atomic num {atomic_num}:', all_coordinates.shape)
    return all_coordinates, confIds, atomIds, molIds


def cluster_coordinates(coordinates, dist_threshold=0.5):
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=None,
                                    linkage='average',
                                    distance_threshold=dist_threshold).fit(coordinates)
    labels = clustering.labels_
    n_clusters = clustering.n_clusters_
    centroid_coords = []
    for label in tqdm(range(n_clusters), total=n_clusters):
        coords = np.array([coord for coord, lab in zip(coordinates, labels) if lab == label])
        mean_coord = coords.mean(axis=0)
        centroid_coords.append(mean_coord)

    return centroid_coords, labels, n_clusters


fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)


def extract_h_bonders(mol):
    feats = factory.GetFeaturesForMol(mol)
    donor_atoms = []
    acceptor_atoms = []
    for feat in feats:
        if feat.GetFamily() == 'Donor':
            donor_atoms.extend(feat.GetAtomIds())
        if feat.GetFamily() == 'Acceptor':
            acceptor_atoms.extend(feat.GetAtomIds())

    return donor_atoms, acceptor_atoms


def get_all_hbonders(mols):
    is_donor = []
    is_acceptor = []
    for i, mol in tqdm(enumerate(mols), total=len(mols), position=0, leave=True):
        conf = mol.GetConformer()
        donor_atoms, acceptor_atoms = extract_h_bonders(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 0:
                idx = atom.GetIdx()
                if idx in donor_atoms:
                    is_donor.append(True)
                else:
                    is_donor.append(False)
                if idx in acceptor_atoms:
                    is_acceptor.append(True)
                else:
                    is_acceptor.append(False)
    return is_donor, is_acceptor

def get_distance(coord1, coord2):
    sq = (coord1 - coord2) ** 2
    return np.sqrt(np.sum(sq))


def save_close_protein_atoms(pdb_file, ligand_coords, dist_threshold=8, output_fname=None, save_pocket_file=None):
    """
    Get the coordinates of 'pocket atoms' (within a certain distance) to limit the number of distances that have to
    be calculated later. Optionally save as a standalone pdb file.

    :param pdb_file:
    :param ligand_coords:
    :param dist_threshold:
    :param output_fname:
    :param savePocketFile:
    :return:
    """
    from pymol import cmd
    coords = {"coords": [], "IDs": [], "elems": []}
    cmd.reinitialize()
    cmd.load(pdb_file, "prot")
    cmd.iterate_state(1, "all", "coords.append([x,y,z])", space=coords)
    cmd.iterate_state(1, "all", "IDs.append(ID)", space=coords)
    cmd.iterate_state(1, "all", "elems.append(elem)", space=coords)

    # select those protein atoms that are within a certain distance and angle of the exit vector
    pocket_coords = {"coords": [], "IDs": [], "elems": []}
    for prot_coord, id, elem in zip(coords["coords"], coords["IDs"], coords["elems"]):
        if id not in pocket_coords["IDs"]:
            for ligand_coord in ligand_coords:
                if id not in pocket_coords["IDs"]:
                    dist = get_distance(ligand_coord, np.array(prot_coord))
                    # print(dist, dist_threshold)
                    if dist <= dist_threshold:
                        pocket_coords["IDs"].append(id)
                        pocket_coords["coords"].append(prot_coord)
                        pocket_coords["elems"].append(elem)

    # save a pdb file with just the selected protein atoms
    if save_pocket_file:
        if len(pocket_coords["IDs"]) > 0:
            select_IDs = pocket_coords["IDs"]
            ids_string = "id "
            for num in select_IDs[: len(select_IDs) - 1]:
                ids_string += str(num)
                ids_string += "+"
            ids_string += str(select_IDs[-1])
            cmd.select("pocket", ids_string)
            cmd.save(output_fname, "pocket")

    return coords, pocket_coords



def get_coords_from_pdb(pdb_file):
    from pymol import cmd
    coords = {"coords": [], "IDs": [], "elems": []}
    cmd.reinitialize()
    cmd.load(pdb_file, "prot")
    cmd.iterate_state(1, "all", "coords.append([x,y,z])", space=coords)
    cmd.iterate_state(1, "all", "IDs.append(ID)", space=coords)
    cmd.iterate_state(1, "all", "elems.append(elem)", space=coords)
    return coords



def get_all_coords(mol):
    conf = mol.GetConformer()
    coords = [np.array(conf.GetAtomPosition(idx)) for idx in range(mol.GetNumAtoms())]
    return coords


def get_all_coords_and_molnums(mol):
    # from config import atom_radii
    atom_radii = {
        6: 1.70,
        7: 1.55,
        8: 1.52,
        16: 1.80,
        9: 1.47,
        15: 1.80,
        17: 1.75,
        12: 1.73,
        35: 1.85
    }
    conf = mol.GetConformer()
    coords = [np.array(conf.GetAtomPosition(idx)) for idx in range(mol.GetNumAtoms())]
    molnums = [mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in range(mol.GetNumAtoms())]
    radii = [atom_radii[num] for num in molnums]
    return coords, radii