
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from scipy.spatial.transform import Rotation as R


def get_anchor_mol():
    anchor_mol = Chem.MolFromSmiles('CC')
    AllChem.EmbedMolecule(anchor_mol, randomSeed=1)
    conf = anchor_mol.GetConformer()
    at1_pos, at2_pos = np.array(conf.GetAtomPosition(0)), np.array(conf.GetAtomPosition(1))
    return at1_pos, at2_pos


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


def as_point3D(coords):
    return Point3D(float(coords[0]),
        float(coords[1]),
        float(coords[2]))


def align_conf(m, confId, ref_attach_coords, ref_adj_coords, return_mol=False):
    if return_mol:  # if we are only performing alignment on a mol with single conformer (confId=0)
        mol = Chem.Mol(m)
    else:  # if we are embedding the original decorators (single mols with multiple conformers)
        mol = m
    attach_idx, adj_idx = get_vector_idxs(mol)
    other_idxs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in [attach_idx, adj_idx]]
    conf = mol.GetConformer(confId)
    attach_coords = np.array(conf.GetAtomPosition(attach_idx))
    adj_coords = np.array(conf.GetAtomPosition(adj_idx))
    other_coords = [np.array(conf.GetAtomPosition(idx)) for idx in other_idxs]

    # ref_attach_coords, ref_adj_coords = get_anchor_mol()
    translation = calc_translation(ref_adj_coords, adj_coords)
    adj_coords = ref_adj_coords
    attach_coords = translate_coords(translation, attach_coords)
    other_coords = [translate_coords(translation, coords) for coords in other_coords]
    # print(ref_adj_coords, ref_attach_coords)
    rotat_mat = get_rotation_matrix_for_mol(ref_adj_coords, ref_attach_coords, attach_coords)
    attach_coords = get_rotated_coord(adj_coords, attach_coords, rotat_mat)
    other_coords = [get_rotated_coord(adj_coords, coords, rotat_mat) for coords in other_coords]

    conf.SetAtomPosition(adj_idx, as_point3D(adj_coords))
    conf.SetAtomPosition(attach_idx, as_point3D(attach_coords))
    for coords, idx in zip(other_coords, other_idxs):
        conf.SetAtomPosition(idx, as_point3D(coords))

    if return_mol:
        return mol


def get_rotated_coord(ref_coordA, coord, rotat):
    query_vec = coord - ref_coordA
    new_coord = ref_coordA + rotat.dot(query_vec)
    return new_coord


def translate_coords(translation, coords):
    return coords + translation


def calc_translation(coord_ref, coord_query):
    return coord_ref - coord_query


def get_rotation_matrix(vec2, vec1=np.array([1, 0, 0])):
    """Get rotation matrix between two vectors using scipy"""
    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))
    r = R.align_vectors(vec2, vec1)
    # print([i for i in r])
    return r[0].as_matrix()


def get_rotation_matrix_for_mol(ref_coordA, ref_coordB, query_coordB):
    ref_vec = ref_coordB - ref_coordA
    query_vec = query_coordB - ref_coordA
    rotat = get_rotation_matrix(ref_vec, query_vec)
    return rotat


def embed(smi, num_confs):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    confIds = AllChem.EmbedMultipleConfs(mol, num_confs)
    mol = Chem.RemoveHs(mol)
    # print(list(confIds))
    ref_attach_coords, ref_adj_coords = get_anchor_mol()
    for confId in confIds:
        align_conf(mol, confId, ref_attach_coords, ref_adj_coords)
    return mol, confIds

