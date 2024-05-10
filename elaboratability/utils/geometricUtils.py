import os

import elaboratability.utils.geometricConfig as config
import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit.Geometry import Point3D
from scipy.spatial.transform import Rotation as R


def as_point3D(coords):
    """

    :param coords:
    :return:
    """
    return Point3D(float(coords[0]),
        float(coords[1]),
        float(coords[2]))


def get_rotated_coord(ref_coordA, coord, rotat):
    """

    :param ref_coordA:
    :param coord:
    :param rotat:
    :return:
    """
    query_vec = coord - ref_coordA
    new_coord = ref_coordA + rotat.dot(query_vec)
    return new_coord


def translate_coords(translation, coords):
    """

    :param translation:
    :param coords:
    :return:
    """
    return coords + translation


def calc_translation(coord_ref, coord_query):
    """

    :param coord_ref:
    :param coord_query:
    :return:
    """
    return coord_ref - coord_query


def get_rotation_matrix(vec2, vec1=np.array([1, 0, 0])):
    """Get rotation matrix between two vectors using scipy"""
    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))
    r = R.align_vectors(vec2, vec1)
    # print([i for i in r])
    return r[0].as_matrix()

def get_anchor_mol():
    """
    Get atom coordinates for a reference anchor mol used throughout

    :return:
    """
    anchor_mol = Chem.MolFromSmiles('CC')
    AllChem.EmbedMolecule(anchor_mol, randomSeed=1)
    conf = anchor_mol.GetConformer()
    at1_pos, at2_pos = np.array(conf.GetAtomPosition(0)), np.array(conf.GetAtomPosition(1))
    return at1_pos, at2_pos


def get_rotation_matrix_for_mol(ref_coordA, ref_coordB, query_coordB):
    """

    :param ref_coordA:
    :param ref_coordB:
    :param query_coordB:
    :return:
    """
    ref_vec = ref_coordB - ref_coordA
    query_vec = query_coordB - ref_coordA
    rotat = get_rotation_matrix(ref_vec, query_vec)
    return rotat


def extract_h_bonders(mol):
    """

    :param mol:
    :return:
    """
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


def get_ligand_data(mol, atom_radii=config.ATOM_RADII):
    """

    :param mol:
    :param atom_radii:
    :return:
    """
    conf = mol.GetConformer()
    ligand_data = {}
    donors, acceptors = extract_h_bonders(mol)
    for idx in range(mol.GetNumAtoms()):
        ligand_data[idx] = {'coords': np.array(conf.GetAtomPosition(idx)),
                            'radii': atom_radii[mol.GetAtomWithIdx(idx).GetAtomicNum()],
                            'is_don': idx in donors,
                            'is_acc': idx in acceptors}
    return ligand_data



def get_all_coords_and_radii(mol, atom_radii=config.ATOM_RADII):
    """

    :param mol:
    :return:
    """
    conf = mol.GetConformer()
    coords = [np.array(conf.GetAtomPosition(idx)) for idx in range(mol.GetNumAtoms())]
    molnums = [mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in range(mol.GetNumAtoms())]
    radii = [atom_radii[num] for num in molnums]
    return coords, radii


def get_hydrogen_vector_pairs(mol, addHs=True):
    """
    Get vector indices for H atom vectors

    :param mol:
    :param addHs:
    :return:
    """
    h_idxs_pairs = []
    if addHs:
        mol = Chem.AddHs(mol, addCoords=True)

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            h_neighbour = atom.GetNeighbors()[0].GetIdx()
            h_idxs_pairs.append([h_neighbour, atom.GetIdx()])

    if addHs:
        return mol, h_idxs_pairs

    else:
        return h_idxs_pairs


def check_terminal_atom(atom):
    """

    :param atom:
    :return:
    """
    atom_neighbour_pairs = []
    atom_idx = atom.GetIdx()

    neighbours = [neigh for neigh in atom.GetNeighbors() if neigh.GetAtomicNum() != 1]
    for neighbour in neighbours:
        neighbour_neighbours = [neighbour_neighbour for neighbour_neighbour in neighbour.GetNeighbors() if
                                neighbour_neighbour.GetAtomicNum() != 1 and neighbour_neighbour.GetIdx() != atom_idx]

        if len(neighbour_neighbours) == 0:
            atom_neighbour_pairs.append([atom_idx, neighbour.GetIdx()])

    return atom_neighbour_pairs


def get_non_hydrogen_vector_pairs(mol):
    """

    :param mol:
    :return:
    """
    vector_pairs = []

    for atom in mol.GetAtoms():
        pairs = check_terminal_atom(atom)
        if len(pairs) > 0:
            vector_pairs.extend(pairs)

    return vector_pairs


def get_coords_of_vector(mol, idx1, idx2):
    """

    :param mol:
    :param idx1:
    :param idx2:
    :return:
    """
    conf = mol.GetConformer()
    coords1 = conf.GetAtomPosition(idx1)
    coords2 = conf.GetAtomPosition(idx2)
    return np.array(coords1), np.array(coords2)


def align_cloud_to_vector(h_coords, neigh_coords, cloud_coords, cloud_attach_coords, cloud_adj_coords):
    """

    :param h_coords:
    :param neigh_coords:
    :param cloud_coords:
    :param cloud_attach_coords:
    :param cloud_adj_coords:
    :return:
    """
    # attach coords and adj coords should just be the anchor mol coordinates used to generate the cloud
    translation = calc_translation(h_coords, cloud_adj_coords)  # ref coord, query coord
    cloud_adj_coords = h_coords

    cloud_attach_coords = translate_coords(translation, cloud_attach_coords)
    all_coords = [translate_coords(translation, coords) for coords in cloud_coords]
    rotat_mat = get_rotation_matrix_for_mol(h_coords, neigh_coords, cloud_attach_coords)
    all_coords = [get_rotated_coord(cloud_adj_coords, coords, rotat_mat) for coords in all_coords]
    return all_coords


def rotate_h_coords(h_coords, anchor_coord, angle, n_rotations):
    """

    :param h_coords:
    :param anchor_coord:
    :param angle:
    :param n_rotations:
    :return:
    """
    avg_coord = np.average(h_coords, axis=0)
    axis_vector = avg_coord - anchor_coord
    axis = axis_vector / np.linalg.norm(axis_vector)

    all_new_h_coords = []

    def rotate_coords(h_coords, anchor_coord, angle):
        """

        :param h_coords:
        :param anchor_coord:
        :param angle:
        :return:
        """
        new_h_coords = []
        for h_coord in h_coords:
            h_vector = h_coord - anchor_coord

            # Compute the rotation matrix
            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)
            cross_product_matrix = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            rotation_matrix = np.eye(3) * cos_theta + sin_theta * cross_product_matrix \
                              + (1 - cos_theta) * np.outer(axis, axis)

            # Apply the rotation to the vector
            rotated_vector = np.dot(rotation_matrix, h_vector)
            new_h_coord = anchor_coord + rotated_vector
            new_h_coords.append(new_h_coord)
        return new_h_coords

    starting_coords = h_coords
    for rotation in range(n_rotations):
        new_coords = np.array(rotate_coords(starting_coords, anchor_coord, angle))
        all_new_h_coords.append(new_coords)
        starting_coords = new_coords

    return all_new_h_coords
