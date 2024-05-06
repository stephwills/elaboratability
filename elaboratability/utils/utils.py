import json
import numpy as np
from rdkit import Chem


def get_intersect(lst1, lst2):
    return list(set(lst1) & set(lst2))


def add_atom_labels(mol):
    for atom in mol.GetAtoms():
        atom.SetProp('atomLabel', str(atom.GetIdx()))


def rectify_mol(mol):
    new_mol = Chem.Mol(mol)
    from molecular_rectifier import Rectifier
    recto = Rectifier(new_mol)
    recto.fix()
    return recto.mol


def add_properties_from_dict(mol, properties):
    for property in properties:
        mol.SetProp(property, properties[property])

def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
    return data


def dump_json(data, fname):
    with open(fname, 'w') as f:
        json.dump(data, f)


def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
    return data


def disable_rdlogger():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')


def save_npy(arr, fname):
    with open(fname, 'wb') as f:
        np.save(f, arr)


def get_bool_intersect(lst1, lst2):
    return [a and b for a, b in zip(lst1, lst2)]


def get_distance(coord1, coord2):
    sq = (coord1 - coord2) ** 2
    return np.sqrt(np.sum(sq))


def set_og_idxs(mol, prop_name):
    """

    :param mol:
    :param prop_name:
    :return:
    """
    for atom in mol.GetAtoms():
        atom.SetProp(prop_name, str(atom.GetIdx()))


def get_new_idx(mol, og_idx, prop_name):
    """

    :param mol:
    :param og_idx:
    :param prop_name:
    :return:
    """
    new_idx = None
    for atom in mol.GetAtoms():
        if atom.HasProp(prop_name):
            if atom.GetProp(prop_name) == str(og_idx):
                new_idx = atom.GetIdx()

    return new_idx
