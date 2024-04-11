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