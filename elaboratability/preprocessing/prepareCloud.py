import os

import elaboratability.utils.processConfig as config
from elaboratability.utils.utils import load_json, disable_rdlogger
from rdkit import Chem
import numpy as np
from tqdm import tqdm


disable_rdlogger()

def load_descriptors(decorator_file=config.DECORATOR_FILE, num_counts_filter=config.MIN_DECORATOR_COUNTS):
    """
    Load decorators from file and filter according to how often they appear in different molecules (extracted from
    ChEMBL).

    :param decorator_file:
    :param num_counts_filter:
    :return:
    """
    decs_dict = load_json(decorator_file)
    select_decs = [k for k, v in decs_dict.items() if v >= num_counts_filter]
    mols = [Chem.MolFromSmiles(s) for s in select_decs]
    return select_decs, mols


def get_all_hbonders(mols):
    is_donor = []
    is_acceptor = []
    for i, mol in tqdm(enumerate(mols), total=len(mols), position=0, leave=True):
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


def extract_h_bonders(mol):
    """

    :param mol:
    :return:
    """
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(mol)
    donor_atoms = []
    acceptor_atoms = []
    for feat in feats:
        if feat.GetFamily() == 'Donor':
            donor_atoms.extend(feat.GetAtomIds())
        if feat.GetFamily() == 'Acceptor':
            acceptor_atoms.extend(feat.GetAtomIds())

    return donor_atoms, acceptor_atoms


def get_coordinates_for_elem(mols: list, mol_ids: list, hbond_data: list, atomic_num: int):
    """
    Get coordinates for specific atom type from list of molecules (with associated IDs of multiple conformers for same
    molecule)

    :param mols: list of RDKit molecules
    :param mol_ids: list of associated identifiers
    :param hbond_data: list of tuples with hbond data for each mol
    :param atomic_num: the atomic number of the atom type to retrieve
    :return:
    """
    all_coordinates = []

    conf_ids = []  # record which conformer (individual mol in list)
    atom_ids = []  # record the atom id (rdkit numbering) for mol
    enumerated_mol_ids = []  # record which molecule the coord belongs to
    is_hbond_don = []
    is_hbond_acc = []

    for conf_id, (mol, mol_id, hbond_info) in tqdm(enumerate(zip(mols, mol_ids, hbond_data)), total=len(mols), position=0, leave=True):
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == atomic_num:
                # get index and coordinates
                idx = atom.GetIdx()
                coords = np.array(conf.GetAtomPosition(idx))

                # record data
                all_coordinates.append(coords)
                atom_ids.append(idx)
                conf_ids.append(conf_id)
                enumerated_mol_ids.append(mol_id)

                hbond_dons, hbond_accs = hbond_info[0], hbond_info[1]
                is_hbond_don.append(idx in hbond_dons)
                is_hbond_acc.append(idx in hbond_accs)

    # list to numpy array
    all_coordinates = np.array(all_coordinates)
    print(f'Coords shape for atomic num {atomic_num}:', all_coordinates.shape)
    elem_data = {'coords': all_coordinates,
                 'conf_ids': conf_ids,
                 'atom_ids': atom_ids,
                 'mol_ids': enumerated_mol_ids,
                 'is_don': is_hbond_don,
                 'is_acc': is_hbond_acc}
    return elem_data


def cluster_coordinates(coordinates, clustering_dist_threshold=config.CLUSTERING_DIST_THRESHOLD):
    """
    Cluster coordinates and get centroid positions using agglomerative clustering

    :param coordinates:
    :param dist_threshold:
    :return:
    """
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=None,
                                    linkage='average',
                                    distance_threshold=clustering_dist_threshold).fit(coordinates)
    labels = clustering.labels_
    n_clusters = clustering.n_clusters_
    centroid_coords = []

    # get centroid (mean coordinate)
    for clust in tqdm(range(n_clusters), total=n_clusters):
        coords = np.array([coord for coord, lab in zip(coordinates, labels) if lab == clust])
        mean_coord = coords.mean(axis=0)
        centroid_coords.append(mean_coord)

    return centroid_coords, labels, n_clusters



def get_all_coordinates(mols: list, mol_ids: list, elements=config.ELEMENT_NAMES, atom_dict=config.ATOM_DICT):
    """

    :param mols:
    :param mol_ids:
    :param elements:
    :param atom_dict:
    :return:
    """
    # get hbond data for all the mols first
    print('Extracting hbonder data')
    hbond_data = [extract_h_bonders(mol) for mol in tqdm(mols)]
    print('Hbonds info retrieved')
    atom_nums = [atom_dict[elem]['atomic_number'] for elem in elements]

    # for each cluster - record associated labels for the coordinate labels
    total_n_clusters = 0

    all_elem_data = {}

    for elem, atom_num in zip(elements, atom_nums):
        print('Processing element', elem)
        elem_data = get_coordinates_for_elem(mols, mol_ids, hbond_data, atom_num)
        if len(elem_data['coords']) > 0:
            centroids, labels, n_clusters = cluster_coordinates(elem_data['coords'])
            # labels = [lab + total_n_clusters for lab in labels]
            elem_data['labels'] = labels
            elem_data['centroids'] = centroids
            elem_data['n_clusters'] = n_clusters
            total_n_clusters += n_clusters
            all_elem_data[elem] = elem_data

    return all_elem_data


def prepare_cloud(conf_file=config.CLUSTERED_CONFORMER_FILE, info_file=config.CLUSTERED_CONFORMER_JSON):
    """

    :param conf_file:
    :param info_file:
    :return:
    """
    conformers = [mol for mol in Chem.SDMolSupplier(conf_file)]
    mol_ids = load_json(info_file)['molIds']
    print(len(conformers), 'conformers loaded for', len(set(mol_ids)), 'molecules')

    print('Processing conformers to create cloud')
    data = get_all_coordinates(conformers,
                               mol_ids)


# create cluster objects - have associated coordinates, element type, conformer IDs, mol IDs,
