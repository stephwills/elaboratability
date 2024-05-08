import json
from argparse import ArgumentParser

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Geometry import Point3D
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


####################################################### UTILS #########################################################

def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
    return data


def dump_json(data, fname):
    with open(fname, 'w') as f:
        json.dump(data, f)

################################################ CODE FOR EMBEDDING ###################################################

def get_vector_idxs(mol, asAsterisk=True):
    if asAsterisk:
        attach_idx = mol.GetSubstructMatch(Chem.MolFromSmiles('*'))[0]
    else:
        attach_idx = None
        for atom in mol.GetAtoms():
            if atom.HasProp('dummyLabel'):
                attach_idx = atom.GetIdx()
                break
    adj_idx = mol.GetAtomWithIdx(attach_idx).GetNeighbors()[0].GetIdx()
    return attach_idx, adj_idx


def calc_translation(coord_ref, coord_query):
    return coord_ref - coord_query


def translate_coords(translation, coords):
    return coords + translation


def get_rotation_matrix(vec2, vec1=np.array([1, 0, 0])):
    """Get rotation matrix between two vectors using scipy"""
    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))
    r = R.align_vectors(vec2, vec1)
    return r[0].as_matrix()


def get_rotation_matrix_for_mol(ref_coordA, ref_coordB, query_coordB):
    ref_vec = ref_coordB - ref_coordA
    query_vec = query_coordB - ref_coordA
    rotat = get_rotation_matrix(ref_vec, query_vec)
    return rotat


def get_rotated_coord(ref_coordA, coord, rotat):
    query_vec = coord - ref_coordA
    new_coord = ref_coordA + rotat.dot(query_vec)
    return new_coord


def align_conf(m, confId, ref_attach_coords, ref_adj_coords, return_mol=False):
    """
    Align a specific conformer for a molecule to reference atom coordinates.

    :param m:
    :param confId:
    :param ref_attach_coords:
    :param ref_adj_coords:
    :param return_mol:
    :return:
    """
    if return_mol:  # if we are only performing alignment on a mol with single conformer (confId=0)
        mol = Chem.Mol(m)
    else:  # if we are embedding the original decorators (single mols with multiple conformers)
        mol = m

    # get the coordinates of the atoms from the embedding molecule to use for alignment
    attach_idx, adj_idx = get_vector_idxs(mol)
    other_idxs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in [attach_idx, adj_idx]]
    conf = mol.GetConformer(confId)
    attach_coords = np.array(conf.GetAtomPosition(attach_idx))
    adj_coords = np.array(conf.GetAtomPosition(adj_idx))
    other_coords = [np.array(conf.GetAtomPosition(idx)) for idx in other_idxs]

    # translate the molecule to the reference coordinates
    translation = calc_translation(ref_adj_coords, adj_coords)
    adj_coords = ref_adj_coords
    attach_coords = translate_coords(translation, attach_coords)
    other_coords = [translate_coords(translation, coords) for coords in other_coords]

    # rotate the molecule so it now aligns with the reference vector
    rotat_mat = get_rotation_matrix_for_mol(ref_adj_coords, ref_attach_coords, attach_coords)
    attach_coords = get_rotated_coord(adj_coords, attach_coords, rotat_mat)
    other_coords = [get_rotated_coord(adj_coords, coords, rotat_mat) for coords in other_coords]

    conf.SetAtomPosition(adj_idx, as_point3D(adj_coords))
    conf.SetAtomPosition(attach_idx, as_point3D(attach_coords))
    for coords, idx in zip(other_coords, other_idxs):
        conf.SetAtomPosition(idx, as_point3D(coords))

    if return_mol:
        return mol


def get_anchor_mol():
    """
    An anchor molecule used as reference coordinates for aligning all decorators. Coordinates are always the same.

    :return:
    """
    anchor_mol = Chem.MolFromSmiles('CC')
    AllChem.EmbedMolecule(anchor_mol, randomSeed=1)
    conf = anchor_mol.GetConformer()
    at1_pos, at2_pos = np.array(conf.GetAtomPosition(0)), np.array(conf.GetAtomPosition(1))
    return at1_pos, at2_pos


def embed(smi: str, num_confs: int):
    """
    Embed the molecule and align all conformers to the same reference point. Return the molecule and the list of
    conformer IDs.

    :param smi:
    :param num_confs:
    :return:
    """
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    confIds = AllChem.EmbedMultipleConfs(mol, num_confs)
    mol = Chem.RemoveHs(mol)
    ref_attach_coords, ref_adj_coords = get_anchor_mol()
    for confId in confIds:
        align_conf(mol, confId, ref_attach_coords, ref_adj_coords)
    return mol, confIds


def embed_and_align_to_anchor(smi: str, n_rotat: int):
    """
    Generate several embeddings for each decorator SMILES according to the number of rotatable bonds. Return the
    embedded molecule (has multiple conformers attached) and the number of conformers generated).

    :param smi:
    :param n_rotat:
    :return:
    """
    n_conf = None
    if n_rotat <= 7:
        n_conf = 50
    if 8 <= n_rotat <= 12:
        n_conf = 200
    if n_rotat > 12:
        n_conf = 300
    emb, _ = embed(smi, n_conf)
    return emb, n_conf

################################################ CODE FOR ROTATIONS ###################################################

def rotate_points(points, axis, angle):
    """

    :param points:
    :param axis:
    :param angle:
    :return:
    """
    # Ensure axis is a unit vector
    axis = axis / np.linalg.norm(axis)
    # Compute rotation matrix using Rodrigues' rotation formula
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    cross_product_matrix = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    rotation_matrix = cos_theta * np.eye(3) + sin_theta * cross_product_matrix + (1 - cos_theta) * np.outer(axis, axis)
    # Rotate the points
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points


def get_axis(point1, point2):
    return point1 - point2


def as_point3D(coords):
    return Point3D(float(coords[0]),
        float(coords[1]),
        float(coords[2]))


def get_rotated_coords(mol, confId=0, angleInterval=30):
    """

    :param mol:
    :param confId:
    :return:
    """
    conf = mol.GetConformer(confId)

    dummy_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0][0]
    dummy_coords = np.array(conf.GetAtomPosition(dummy_idx))

    adj_idx = mol.GetAtomWithIdx(dummy_idx).GetNeighbors()[0].GetIdx()
    adj_coords = np.array(conf.GetAtomPosition(adj_idx))

    other_idxs = [idx for idx in range(mol.GetNumAtoms()) if idx not in [dummy_idx, adj_idx]]
    coords = np.array([np.array(conf.GetAtomPosition(idx)) for idx in other_idxs])

    rotat_angles = [i for i in range(0,360,angleInterval)]
    all_coords = []

    all_idxs = other_idxs + [dummy_idx, adj_idx]
    vector_coords = np.array([dummy_coords, adj_coords])
    for _angle in rotat_angles:
        angle = _angle * (np.pi/180)
        new_coords = rotate_points(coords, get_axis(dummy_coords, adj_coords), angle)
        new_coords = np.vstack((new_coords, vector_coords))
        all_coords.append(new_coords)

    return all_idxs, all_coords


def cluster_mol_conformers_wout_alignment(mol, n_cids, distThreshold=1.5):
    """

    :param mol:
    :param n_cids:
    :param distThreshold:
    :return:
    """
    from rdkit.Chem import rdMolAlign
    from rdkit.ML.Cluster import Butina

    cids = list(range(n_cids))
    dists = []
    for i in range(len(cids)):
        for j in range(i):
            # _ = rdMolAlign.AlignMol(mol,mol,prbCid=i,refCid=j)
            dists.append(rdMolAlign.CalcRMS(mol,mol,i,j))
    clusts = Butina.ClusterData(dists, len(cids), distThreshold, isDistData=True, reordering=True)
    return clusts


def cluster_rotations(mol, n_confs, n_rotats=12, distThreshold=1.5, angleInterval=30):
    """

    :param mol:
    :param n_confs:
    :param n_rotats:
    :return:
    """
    if mol.GetNumAtoms() == 2:
        return mol
    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()

    for i in range(n_confs):
        conf = mol.GetConformer(i)
        coord_idxs, new_coords = get_rotated_coords(mol, i, angleInterval)
        for j, coords in enumerate(new_coords):
            new_conf_idx = i*(n_rotats) + j
            new_mol.AddConformer(conf, assignId=new_conf_idx)
            new_conf = new_mol.GetConformer(new_conf_idx)
            for idx, coord in zip(coord_idxs, coords):
                new_conf.SetAtomPosition(idx, as_point3D(coord))

    clustered_mol = Chem.Mol(mol)
    clustered_mol.RemoveAllConformers()

    clusters = cluster_mol_conformers_wout_alignment(new_mol, n_cids=new_mol.GetNumConformers(), distThreshold=distThreshold)
    for c, cluster in enumerate(clusters):
        centroid = cluster[0]
        add_conf = new_mol.GetConformer(centroid)
        clustered_mol.AddConformer(add_conf, assignId=c)

    # print(new_mol.GetNumConformers(), 'num conformers reduced to', clustered_mol.GetNumConformers())
    return clustered_mol


def main():
    parser = ArgumentParser()
    parser.add_argument('--decs_dict_json', help='a json file containing a list of decorator SMILES')
    parser.add_argument('--threshold_frequency', type=int)
    parser.add_argument('--minimum_atoms', type=int, default=3)
    parser.add_argument('--n_cpus', type=int)
    parser.add_argument('--output_sdf')
    parser.add_argument('--output_json')
    parser.add_argument('--n_rotations', type=int, default=12)
    parser.add_argument('--distThreshold', type=float, default=1.5)
    parser.add_argument('--angle_interval', type=int, default=30)
    args = parser.parse_args()

    # read in decorators and calculate number of rotatable bonds (to decide how many embeddings to create)
    decs_dict = load_json(args.decs_dict_json)
    counts = list(decs_dict.values())
    common_decs = [k for k, v in tqdm(decs_dict.items()) if v >= args.threshold_frequency]
    common_decs = [smi for smi in common_decs if Chem.MolFromSmiles(smi).GetNumAtoms() >= args.minimum_atoms]
    mols = [Chem.MolFromSmiles(s) for s in common_decs]
    rotats = [rdMolDescriptors.CalcNumRotatableBonds(mol) for mol in mols]
    print(len(mols), 'elaborations to be used to create conformers')

    # generate embedded decorators and align them to the same reference point
    results = Parallel(n_jobs=args.n_cpus, backend="multiprocessing")(
        delayed(embed_and_align_to_anchor)(smi, n_rotat) for smi, n_rotat
        in tqdm(zip(common_decs, rotats), total=len(common_decs),
                leave=True, position=0)
    )

    embedded_mols = [res[0] for res in results]
    n_confs = [res[1] for res in results]

    print(len(embedded_mols), 'read')
    print('generating rotations and clustering')

    # rotate each embedded conformer and cluster the conformations to get a representative set of rotated, clustered
    # embeddings
    new_mols = Parallel(n_jobs=args.n_cpus, backend="multiprocessing")(
        delayed(cluster_rotations)(mol, n_conf, args.n_rotations, args.distThreshold, args.angle_interval) for mol, n_conf
        in tqdm(zip(embedded_mols, n_confs), total=len(embedded_mols), leave=True, position=0)
    )

    print('new mols generated')
    print('writing sdf')

    # write the mols to file
    all_ids = []
    all_mol_ids = []
    all_smi = []
    w = Chem.SDWriter(args.output_sdf)
    for i, (mol, dec) in enumerate(zip(new_mols, common_decs)):
        ids = []
        for conf in mol.GetConformers():
            id = conf.GetId()
            w.write(mol, confId=id)
            ids.append(id)
            all_mol_ids.append(i)
            all_smi.append(dec)

        all_ids.append(ids)
    w.close()
    print('sdf written')

    d = {'confIds': all_ids,
         'molIds': all_mol_ids,
         'smis': all_smi}
    dump_json(d, args.output_json)
    print('json written')


if __name__ == "__main__":
    main()
