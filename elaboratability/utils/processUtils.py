from rdkit import Chem
from rdkit.Chem import rdFMCS, Mol, AllChem
import elaboratability.utils.processConfig as proConfig


def get_mappings_between_precursor_and_mol(mol: Mol, precursor: Mol):
    """
    Get all possible mappings between a precursor and a molecule it was derived from (accounting for all possible
    substructure matches with the molecule).

    :param mol:
    :param precursor:
    :return:
    """
    import itertools
    mcs = Chem.MolFromSmarts(rdFMCS.FindMCS([mol, precursor], ringMatchesRingOnly=True).smartsString)
    mol_substruct_matches = mol.GetSubstructMatches(mcs)
    precursor_substruct_match = precursor.GetSubstructMatch(mcs)

    maps = []

    for mol_substruct_match in mol_substruct_matches:
        map = {}
        for mol_idx, pre_idx in zip(mol_substruct_match, precursor_substruct_match):
            map[mol_idx] = pre_idx
        maps.append(map)
    return maps, mcs


def get_map_info_from_mol(mol: Mol) -> dict:
    """
    Get Morgan Fingerprint hashing for mapped atoms

    :param mol:
    :return:
    """
    Chem.SanitizeMol(mol)
    bitInfo = {}
    fp = AllChem.GetMorganFingerprint(mol, radius=3, bitInfo=bitInfo)
    bitIds = list(bitInfo.keys())

    mapNumbers = [atom.GetProp('molAtomMapNumber') for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
    mapInfo = {key: {0: None,
                     1: None,
                     2: None,
                     3: None} for key in mapNumbers}

    for bitId in bitIds:
        for tup in bitInfo[bitId]:
            atom_idx, radius = tup[0], tup[1]
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.HasProp('molAtomMapNumber'):
                mapNumber = atom.GetProp('molAtomMapNumber')
                mapInfo[mapNumber][radius] = bitId

    return mapInfo


def get_vector_atoms(mol, precursor, map):
    """

    :param mol:
    :param precursor:
    :param map:
    :return:
    """
    mol, precursor = Chem.Mol(mol), Chem.Mol(precursor)
    mol_substruct_match = list(map.keys())

    vectors_in_mol = []
    vectors = []
    vector_neighbours_in_precursor = []
    vector_types = []

    for mol_at in mol_substruct_match:
        vector_type = None
        # get which molecule is a vector atom by checking which has a neighbour not in the MCS
        mol_vector_neighbours = [atom.GetIdx() for atom in mol.GetAtomWithIdx(mol_at).GetNeighbors() if atom.GetIdx() not in mol_substruct_match]

        # if yes, then check that it is well placed in the precursor (attached to terminal atom or hydrogen to form vector)
        if len(mol_vector_neighbours) > 0:

            vector = map[mol_at]  # get the equivalent vector atom in the precursor

            # record the idxs of the neighbours to use (empty if hydrogen)
            filt_neighbours = []
            vector_check = False

            # get neighbouring atoms to vector in the precursor
            neighbours = [at.GetIdx() for at in precursor.GetAtomWithIdx(vector).GetNeighbors()]
            vector_neighbours = [idx for idx in neighbours if idx not in map.values()]

            # if no neighbours, vector passes
            if len(vector_neighbours) == 0:
                vector_check = True
                vector_type = 'hydrogen'

            # else, check that attached neighbours are terminal atoms
            else:
                for vector_neighbour in vector_neighbours:  # only neighbour should be the original vector atom
                    vector_neighbour_neighbours = [at.GetIdx() for at in precursor.GetAtomWithIdx(vector_neighbour).GetNeighbors() if at.GetIdx() != vector]
                    if len(vector_neighbour_neighbours) == 0:
                        filt_neighbours.append(vector_neighbour)
                        vector_check = True
                        vector_type = 'non-hydrogen'

            if vector_check:
                vectors.append(vector)
                vectors_in_mol.append(mol_at)
                vector_neighbours_in_precursor.append(filt_neighbours)
                vector_types.append(vector_type)

    return vectors, vectors_in_mol, vector_neighbours_in_precursor, vector_types


def constrained_embed_of_precursor(mol: Mol, precursor: Mol, mcs: Mol, map: dict):
    """
    Perform constrained embedding of the precursor using the MCS with the original molecule

    :param mol:
    :param precursor:
    :param mcs:
    :param map:
    :return:
    """
    mol, precursor, mcs = Chem.Mol(mol), Chem.Mol(precursor), Chem.Mol(mcs)

    # try sanitize MCS and generate random embedding
    try:
        Chem.SanitizeMol(mcs)
        AllChem.EmbedMolecule(mcs, randomSeed=42)
    except:
        print('Could not sanitize mcs')
        return None

    # generate random embedding for the precursor
    AllChem.EmbedMolecule(precursor, randomSeed=42)

    # use mcs as template -> set atom positions using mol coordinates
    mcs_conf, mol_conf = mcs.GetConformer(), mol.GetConformer()
    mcs_match = mcs.GetSubstructMatch(mcs)

    for mcs_at, mol_at in zip(mcs_match, map):
        mcs_conf.SetAtomPosition(mcs_at, mol_conf.GetAtomPosition(mol_at))

    mcs.AddConformer(mcs_conf)
    precursor = Chem.AddHs(precursor)

    try:
        AllChem.ConstrainedEmbed(precursor, mcs, randomseed=42)  # TODO: check this is using the correct MCS conf?
    except:
        print('Could not constrained embed')
        return None
    precursor = Chem.RemoveHs(precursor)
    return precursor


def check_elaboration_is_useful(mol: Mol, vector: int, map: dict, int_idxs: list, dists: dict):
    """

    :param mol:
    :param vector:
    :param map:
    :param int_idxs:
    :param dists:
    :return:
    """
    from elaboratability.utils.utils import get_intersect
    mol_mcs_matches = list(map.keys())
    mol_atoms = [i for i in range(mol.GetNumAtoms())]

    def get_elab_idxs(mol, vector, mcs_matches):
        mol = Chem.Mol(mol)
        # retrieve the idxs of atoms that are part of the elaboration
        # get neighbor atoms
        vector_neighbours = [atom.GetIdx() for atom in mol.GetAtomWithIdx(vector).GetNeighbors() if atom.GetIdx() not in mcs_matches]
        bonds = [mol.GetBondBetweenAtoms(vector, neighbour).GetIdx() for neighbour in vector_neighbours]

        # fragment on the vector bonds
        fragments = Chem.FragmentOnBonds(mol, bonds)
        fragment_idxs = Chem.GetMolFrags(fragments)  # tuple containing atom idxs of the fragments

        elab_idxs = set()
        for idxs in fragment_idxs:
            if len(get_intersect(idxs, mol_mcs_matches)) == 0:
                for idx in idxs:
                    if idx in mol_atoms:
                        elab_idxs.add(idx)

        return list(elab_idxs)

    elab_idxs = get_elab_idxs(mol, vector, mol_mcs_matches)
    if len(elab_idxs) == 0:
        return False

    if len(get_intersect(elab_idxs, int_idxs)) > 0:
        return True
    else:
        elab_dists = [dists[idx] for idx in elab_idxs]
        mcs_dists = [dists[idx] for idx in mol_mcs_matches]
        if min(elab_dists) >= min(mcs_dists):
            return True
        else:
            return False


def get_all_coords_from_rdkit(mol: Mol) -> list:
    """

    :param mol:
    :return:
    """
    import numpy as np
    conf = mol.GetConformer()
    coords = [np.array(conf.GetAtomPosition(idx)) for idx in range(mol.GetNumAtoms())]
    return coords


def get_coords_from_pdb(pdb_file: str) -> dict:
    """

    :param pdb_file:
    :return:
    """
    from pymol import cmd
    coords = {"coords": [], "IDs": [], "elems": []}
    cmd.reinitialize()
    cmd.load(pdb_file, "prot")
    cmd.iterate_state(1, "all", "coords.append([x,y,z])", space=coords)
    cmd.iterate_state(1, "all", "IDs.append(ID)", space=coords)
    cmd.iterate_state(1, "all", "elems.append(elem)", space=coords)
    return coords

def get_coords_from_pdb_mdanalysis(pdb_file: str):
    """
    Get coords using mdanalysis as numpy array

    :param pdb_file:
    :return:
    """
    import MDAnalysis
    u = MDAnalysis.Universe(pdb_file)
    receptor = u.select_atoms("protein")
    coords = receptor.positions
    return coords


def get_ph4_atom_ids_from_rdkit(mol: Mol) -> list:
    """

    :param mol:
    :return:
    """
    import os
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    interacting_ids = [feat.GetAtomIds()[0] for feat in factory.GetFeaturesForMol(mol) if
                           feat.GetFamily() == 'Acceptor' or feat.GetFamily() == 'Donor']
    return interacting_ids


def get_ph4_atom_ids_from_pdb(pdb_file: str) -> list:
    """

    :param pdb_file:
    :return:
    """
    from elaboratability.utils.pymolUtils import findSurfaceAtoms, findHDonorsAcceptors
    from elaboratability.utils.utils import get_intersect

    surface_ids = findSurfaceAtoms(pdb_file)['IDs']
    hbonder_data = findHDonorsAcceptors(pdb_file)
    hbond_donor_ids = hbonder_data["donor_IDs"]
    hbond_acceptor_ids = hbonder_data["acceptor_IDs"]
    interacting_ids = get_intersect(hbond_donor_ids, hbond_acceptor_ids)
    interacting_ids = get_intersect(interacting_ids, surface_ids)
    return interacting_ids


def get_possible_interacting_atoms_and_min_dists(mol: Mol, pdb_file: str, hbond_cutoff: float = 3.0, use_pymol=proConfig.USE_PYMOL):
    """

    :param mol:
    :param pdb_file:
    :param hbond_cutoff:
    :return:
    """
    import numpy as np
    from scipy.spatial.distance import cdist
    mol_coords = get_all_coords_from_rdkit(mol)
    if use_pymol:
        prot_coords = get_coords_from_pdb(pdb_file)["coords"]
    else:
        prot_coords = get_coords_from_pdb_mdanalysis(pdb_file)

    dists = cdist(mol_coords, prot_coords)

    # get potential interacting atoms
    potential_mol_interacting_ids = get_ph4_atom_ids_from_rdkit(mol)
    protein_interacting_ids = get_ph4_atom_ids_from_pdb(pdb_file)

    mol_interacting_ids = []

    hbond_dists = np.where(dists <= hbond_cutoff)
    for mol_idx, prot_idx in zip(hbond_dists[0], hbond_dists[1]):
        if mol_idx in potential_mol_interacting_ids and prot_idx in protein_interacting_ids:
            mol_interacting_ids.append(mol_idx)

    # get the closest distance to a protein atom for each ligand atom
    closest_dists = {idx: np.min(dists[idx]) for idx in range(len(mol_coords))}

    return mol_interacting_ids, closest_dists


def check_vector_leads_to_elab_of_size(mol, mcs_atoms, vector, min_ats=3):
    """
    Select elaboratable vectors by checking that they lead to an elaboration of a specific size (i.e. not just the
    addition of one atom)

    :param mol:
    :param mcs_atoms:
    :param vector:
    :param min_ats:
    :return:
    """
    from elaboratability.utils.utils import get_intersect
    og_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
    vector_atom = mol.GetAtomWithIdx(vector)
    neighbours = [i for i in vector_atom.GetNeighbors() if i.GetAtomicNum() != 1 and i.GetIdx() not in mcs_atoms]
    vector_check = False
    for neigh in neighbours:
        neigh_idx = neigh.GetIdx()
        bond = mol.GetBondBetweenAtoms(vector, neigh_idx)
        fragments = Chem.FragmentOnBonds(mol, [bond.GetIdx()])
        fragment_idxs = Chem.GetMolFrags(fragments)
        relevant_fragments = [idxs for idxs in fragment_idxs if len(get_intersect(idxs, mcs_atoms)) == 0]
        for fragment in relevant_fragments:
            n_ats = len(get_intersect(og_atoms, fragment))
            if n_ats >= min_ats:
                vector_check = True

    return vector_check