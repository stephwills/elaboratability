
import numpy as np
from alignConformersToRef import align_conf, get_vector_idxs
from rdkit import Chem


def load_mols_from_sdf(sdf):
    mols = list(Chem.SDMolSupplier(sdf))
    print(len(mols), 'mols loaded')
    return mols


def get_h_idxs(mol, addHs=True):
    if addHs:
        mol = Chem.AddHs(mol, addCoords=True)
    h_idxs = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            h_idxs.append(atom.GetIdx())
    return mol, h_idxs


def get_coords_of_vector(mol, idx):
    conf = mol.GetConformer()
    h_coords = conf.GetAtomPosition(idx)
    neigh_idx = mol.GetAtomWithIdx(idx).GetNeighbors()[0].GetIdx()
    neigh_coords = conf.GetAtomPosition(neigh_idx)
    return np.array(h_coords), np.array(neigh_coords)


def align_elab_to_mol(mol, decorator, attach_idx):
    mol = Chem.Mol(mol)
    h_coords, neigh_coords = get_coords_of_vector(mol, attach_idx)
    # print(h_coords, neigh_coords)
    embedded_decorator = align_conf(decorator, 0, neigh_coords, h_coords, return_mol=True)
    return embedded_decorator


def add_decorator_to_mol(mol, decorator, addHs=True, returnMols=False, writeMols=False):
    mol, h_idxs = get_h_idxs(mol, addHs=addHs)
    # print(h_idxs)
    embedded_decorators = []
    for h_idx in h_idxs:
        # print(h_idx)
        embedded_decorator = align_elab_to_mol(mol, decorator, h_idx)
        embedded_decorator.SetProp('hydrogenAtomIdx', str(h_idx))
        embedded_decorators.append(embedded_decorator)

    if writeMols:
        w = Chem.SDWriter(writeMols)
        for embed in embedded_decorators:
            w.write(embed)
        w.close()

    if returnMols:
        return embedded_decorators


def add_all_decorators_to_idx(mol, decorators, h_idx):
    embedded_decorators = []
    for decorator in decorators:
        embedded_decorator = align_elab_to_mol(mol, decorator, h_idx)
        embedded_decorators.append(embedded_decorator)
    return embedded_decorators


def add_decorators_to_mol(mol, decorators, addHs=True, returnMols=False, writeMols=False):
    all_decorators = []
    for decorator in decorators:
        embedded_decorators = add_decorator_to_mol(mol, decorator, addHs=addHs, returnMols=True)
        all_decorators.extend(embedded_decorators)

    if writeMols:
        w = Chem.SDWriter(writeMols)
        for dec in all_decorators:
            w.write(dec)
        w.close()

    if returnMols:
        return all_decorators
