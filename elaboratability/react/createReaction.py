
import elaboratability.utils.reactConfig as config
from aizynthfinder.chem.mol import TreeMolecule
from aizynthfinder.chem.reaction import SmilesBasedRetroReaction
from elaboratability.utils.utils import get_new_idx, set_og_idxs
from rdkit import Chem
from tqdm import tqdm


def replace_hydrogen_with_decorator(mol, anchor_atom, replace_atom, dec_smi):
    """

    :param mol:
    :param anchor_atom:
    :param dec_smi:
    :return:
    """
    set_og_idxs(mol, 'og_mol_idx')

    dec = Chem.MolFromSmiles(dec_smi)
    dec = Chem.AddHs(dec)
    dec_attach_atom = None
    for atom in dec.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dec_attach_atom = atom.GetIdx()
            atom.SetAtomicNum(1)
    dec_anchor_atom = dec.GetAtomWithIdx(dec_attach_atom).GetNeighbors()[0].GetIdx()

    set_og_idxs(dec, 'og_dec_idx')

    comb_mol = Chem.CombineMols(mol, dec)
    editable_mol = Chem.EditableMol(comb_mol)

    new_mol_anchor = get_new_idx(comb_mol, anchor_atom, 'og_mol_idx')
    new_dec_anchor = get_new_idx(comb_mol, dec_anchor_atom, 'og_dec_idx')
    editable_mol.AddBond(new_mol_anchor, new_dec_anchor, order=Chem.BondType.SINGLE)
    new_mol_replace = get_new_idx(comb_mol, replace_atom, 'og_mol_idx')
    new_dec_replace = get_new_idx(comb_mol, dec_attach_atom, 'og_dec_idx')
    replace = [new_mol_replace, new_dec_replace]
    replace.sort(reverse=True)
    editable_mol.RemoveAtom(replace[0])
    editable_mol.RemoveAtom(replace[1])
    new_mol = editable_mol.GetMol()

    return new_mol


def replace_heavy_atom_with_decorator(mol, anchor_atom, replaced_atom, dec_smi):
    """

    :param mol:
    :param anchor_atom:
    :param replaced_atom:
    :param dec_smi:
    :return:
    """
    intermed_mol, new_anchor_idx, new_replace_idx = switch_replaced_atom_with_hydrogen(mol, anchor_atom, replaced_atom)
    new_mol = replace_hydrogen_with_decorator(intermed_mol, new_anchor_idx, new_replace_idx, dec_smi)
    return new_mol


def switch_replaced_atom_with_hydrogen(mol, anchor_atom, replaced_atom):
    """

    :param mol:
    :param anchor_atom:
    :param replaced_atom:
    :return:
    """
    mol = Chem.Mol(mol)
    set_og_idxs(mol, 'mol_og_idx')
    replaced_at = mol.GetAtomWithIdx(replaced_atom)
    replaced_at.SetAtomicNum(1)
    bond = mol.GetBondBetweenAtoms(anchor_atom, replaced_atom)
    bond_type = bond.GetBondTypeAsDouble()
    h_1 = Chem.MolFromSmiles('[H]')
    h_2 = Chem.MolFromSmiles('[H]')
    set_og_idxs(h_1, 'h1_og_idx')
    set_og_idxs(h_2, 'h2_og_idx')

    if bond_type == 1.0:
        neighs = replaced_at.GetNeighbors()
        neigh_idxs = [at.GetIdx() for at in neighs if at.GetIdx() != anchor_atom]
        neigh_idxs.sort(reverse=True)
        edit = Chem.EditableMol(mol)
        for neigh_idx in neigh_idxs:
            edit.RemoveAtom(neigh_idx)

    if bond_type == 2.0:
        bond.SetBondType(Chem.BondType.SINGLE)
        comb = Chem.CombineMols(mol, h_1)
        new_anchor_idx = get_new_idx(comb, anchor_atom, 'mol_og_idx')
        new_h1_idx = get_new_idx(comb, 0, 'h1_og_idx')
        edit = Chem.EditableMol(comb)
        edit.AddBond(new_anchor_idx, new_h1_idx, order=Chem.BondType.SINGLE)

    if bond_type == 3.0:
        bond.SetBondType(Chem.BondType.SINGLE)
        comb = Chem.CombineMols(mol, h_1)
        comb = Chem.CombineMols(comb, h_2)
        new_anchor_idx = get_new_idx(comb, anchor_atom, 'mol_og_idx')
        new_h1_idx = get_new_idx(comb, 0, 'h1_og_idx')
        new_h2_idx = get_new_idx(comb, 0, 'h2_og_idx')
        edit = Chem.EditableMol(comb)
        edit.AddBond(new_anchor_idx, new_h1_idx, order=Chem.BondType.SINGLE)
        edit.AddBond(new_anchor_idx, new_h2_idx, order=Chem.BondType.SINGLE)

    back = edit.GetMol()
    if bond_type == 1.0:
        new_anchor_idx = get_new_idx(back, anchor_atom, 'mol_og_idx')
        # new_replace_idx = get_new_idx(back, replaced_atom, 'mol_og_idx')
    # else:  # TODO: check if this should be comb???
    new_replace_idx = get_new_idx(back, replaced_atom, 'mol_og_idx')
    return back, new_anchor_idx, new_replace_idx


def remove_attachment_atom_from_decorator(dec_smi):
    """

    :param dec_smi:
    :return:
    """
    dec = Chem.MolFromSmiles(dec_smi)
    dec = Chem.AddHs(dec)
    for atom in dec.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
    dec = Chem.RemoveHs(dec)
    return Chem.MolToSmiles(dec)


def create_reaction_smiles(mol, dec_smi, anchor_atom, replaced_atom, is_hydrogen):
    """

    :param mol:
    :param dec_smi:
    :param anchor_atom:
    :param replaced_atom:
    :param is_hydrogen:
    :return:
    """
    if not is_hydrogen:
        product = replace_heavy_atom_with_decorator(mol, anchor_atom, replaced_atom, dec_smi)
    else:
        product = replace_hydrogen_with_decorator(mol, anchor_atom, replaced_atom, dec_smi)
    mol = Chem.RemoveHs(mol)
    smi = Chem.MolToSmiles(mol)
    product = Chem.RemoveHs(product)
    product_smi = Chem.MolToSmiles(product)
    dec_smi = remove_attachment_atom_from_decorator(dec_smi)
    reactants = f"{smi}.{dec_smi}"
    return reactants, product_smi


def create_reaction_smiles_for_decorators(mol, decorators, anchor_atom, replaced_atom, is_hydrogen):
    """

    :param mol:
    :param decorators:
    :param anchor_atom:
    :param replaced_atom:
    :param is_hydrogen:
    :return:
    """
    all_reactants = []
    all_products = []

    for dec in decorators:
        reactants, product_smi = create_reaction_smiles(mol, dec, anchor_atom, replaced_atom, is_hydrogen)
        all_reactants.append(reactants)
        all_products.append(product_smi)

    return all_reactants, all_products


def filter_single_rxn_with_aizynthfinder(reactant_smi, product_smi):
    """

    :param reactant_smi:
    :param product_smi:
    :return:
    """
    treemol = TreeMolecule(parent=None, smiles=product_smi)
    rxn = SmilesBasedRetroReaction(mol=treemol,
                                   reactants_str=reactant_smi)
    return config.filter.feasibility(rxn)


def filter_multiple_rxns_for_vector_with_aizynthfinder(mol, anchor_atom, replace_atom, vector_is_hydrogen, decorators,
                                            filter_cutoff=None, verbose=False, return_reaction_smiles=False):
    """

    :param mol:
    :param anchor_atom:
    :param replace_atom:
    :param vector_is_hydrogen:
    :param decorators:
    :param filter_cutoff:
    :param verbose:
    :return:
    """
    disable = not verbose
    if filter_cutoff:
        if verbose:
            print('Applying filter cutoff', filter_cutoff)
        config.filter.filter_cutoff = filter_cutoff

    filter_results = []
    filter_feas = []

    all_reactants = []
    all_products = []

    if verbose:
        print(len(decorators), 'decorators to evaluate')
    for decorator in tqdm(decorators, total=len(decorators), position=0, leave=True, disable=disable):
        reacts, prod = create_reaction_smiles(mol, decorator, anchor_atom, replace_atom, vector_is_hydrogen)
        if return_reaction_smiles:
            all_reactants.append(reacts)
            all_products.append(prod)
        treemol = TreeMolecule(parent=None, smiles=prod)
        rxn = SmilesBasedRetroReaction(mol=treemol,
                                       reactants_str=reacts)
        res, feas = config.filter.feasibility(rxn)
        filter_results.append(res)
        filter_feas.append(feas)

    if verbose:
        print(sum(filter_results), 'of', len(filter_results), 'pass filter')

    if not return_reaction_smiles:
        return filter_results, filter_feas
    else:
        return filter_results, filter_feas, all_reactants, all_products


def filter_multiple_rxns_with_aizynthfinder(mol, anchor_atoms, replace_atoms, vectors_are_hydrogens, decorators,
                                            filter_cutoff=None):
    """

    :param mol:
    :param anchor_atoms:
    :param replace_atoms:
    :param vectors_are_hydrogens:
    :param decorators:
    :param filter_cutoff:
    :return:
    """
    if filter_cutoff:
        print('Applying filter cutoff', filter_cutoff)
        config.filter.filter_cutoff = filter_cutoff

    filter_results = []
    filter_feas = []

    for anchor_atom, replace_atom, is_hydrogen, decorator in tqdm(zip(anchor_atoms, replace_atoms, vectors_are_hydrogens, decorators), total=len(decorators), position=0, leave=True):
        reacts, prod = create_reaction_smiles(mol, decorator, anchor_atom, replace_atom, is_hydrogen)
        treemol = TreeMolecule(parent=None, smiles=prod)
        rxn = SmilesBasedRetroReaction(mol=treemol,
                                       reactants_str=reacts)
        res, feas = config.filter.feasibility(rxn)
        filter_results.append(res)
        filter_feas.append(feas)

    print(sum(filter_results), 'of', len(filter_results), 'pass filter')
    return filter_results, filter_feas
