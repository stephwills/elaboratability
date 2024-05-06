
import os
import shutil
import tempfile

from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, rdFMCS, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps
from elaboratability.utils.utils import set_og_idxs, get_new_idx


def save_substructure_to_file(matches, mol, mol_file, pymol_coords, tmp_dir, fname):
    def get_coords_rdkit(atom_ids, mol):
        conf = mol.GetConformer()
        coords = []
        for id in atom_ids:
            coord = conf.GetAtomPosition(id)
            coords.append(list(coord))
        return coords

    def round_coords(coords):
        return [round(c, 0) for c in coords]

    substructure_atom_coords = get_coords_rdkit(matches, mol)
    substructure_atom_coords = [round_coords(coords) for coords in substructure_atom_coords]

    rounded = [round_coords(coords) for coords in pymol_coords["coords"]]
    pymol_coords["coords"] = rounded
    substructure_IDs = []

    for coords in substructure_atom_coords:
        if coords in pymol_coords["coords"]:
            idx = pymol_coords["coords"].index(coords)
            ID = pymol_coords["IDs"][idx]
            substructure_IDs.append(ID)

    def atom_IDs_to_molfile(mol_file, output_file, IDs):
        from pymol import cmd
        cmd.reinitialize()
        cmd.load(mol_file, "sdf")
        ids_string = "id "
        for num in IDs[: len(IDs) - 1]:
            ids_string += str(num)
            ids_string += "+"
        ids_string += str(IDs[-1])
        cmd.select("substructure", ids_string)
        cmd.save(output_file, "substructure")
        return output_file

    if len(substructure_IDs) > 0:
        output_file = atom_IDs_to_molfile(mol_file, os.path.join(tmp_dir, fname), substructure_IDs)
        return output_file

    else:
        return None


def atom_coords_from_pymol(mol_file: str) -> dict:
    """
    Get the coordinates and IDs of all atoms in a molfile

    :param mol_file:
    :return: dictionary with "coords" and "IDs"
    """
    from pymol import cmd
    coords = {"coords": [], "IDs": []}
    cmd.reinitialize()
    cmd.load(mol_file, "sdf")
    cmd.iterate_state(1, "all", "coords.append([x,y,z])", space=coords)
    cmd.iterate_state(1, "all", "IDs.append(ID)", space=coords)
    return coords


def calc_sucos(small_m, large_m):

    score_mode=FeatMaps.FeatMapScoreMode.Best
    fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

    fmParams = {}
    for k in fdef.GetFeatureFamilies():
        fparams = FeatMaps.FeatMapParams()
        fmParams[k] = fparams

    featLists = []
    for m in [small_m, large_m]:
        rawFeats = fdef.GetFeaturesForMol(m)
        featLists.append([f for f in rawFeats])

    fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=fmParams) for x in featLists]
    fms[0].scoreMode = score_mode
    fm_score = fms[0].ScoreFeats(featLists[1]) / min(fms[0].GetNumFeatures(), len(featLists[1]))

    # fm_score = get_FeatureMapScore(reflig, prb_mol, score_mode)
    protrude_dist = rdShapeHelpers.ShapeProtrudeDist(large_m, small_m, allowReordering=False)

    SuCOS_score = 0.5 * fm_score + 0.5 * (1 - protrude_dist)
    return SuCOS_score


def get_vector_from_test_data(fragment_sdf, ligand_sdf):
    """

    :param fragment_sdf:
    :param ligand_sdf:
    :return:
    """
    fragment = Chem.SDMolSupplier(fragment_sdf)[0]
    ligand = Chem.SDMolSupplier(ligand_sdf)[0]

    # get MCS between fragment and ligand
    mcs = Chem.MolFromSmarts(rdFMCS.FindMCS([ligand, fragment],
                                            completeRingsOnly=True,
                                            atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom).smartsString)

    fragment_matches = fragment.GetSubstructMatches(mcs)
    ligand_matches = ligand.GetSubstructMatches(mcs)

    # get the matching atoms with the highest SuCOS overlap
    tmp_dir = tempfile.mkdtemp()
    pymol_lig_coords = atom_coords_from_pymol(ligand_sdf)
    pymol_frag_coords = atom_coords_from_pymol(fragment_sdf)
    lig_files = [save_substructure_to_file(lig_match, ligand, ligand_sdf,
                                           pymol_lig_coords, tmp_dir,
                                           f"lig_{i}.mol") for i, lig_match in enumerate(ligand_matches)]
    fragment_files = [save_substructure_to_file(frag_match, fragment, fragment_sdf,
                                        pymol_frag_coords, tmp_dir,
                                                f"frag_{i}.mol") for i, frag_match in enumerate(fragment_matches)]

    sucoses = []
    idxs = []
    for i, (ligand_match, lig_file) in enumerate(zip(ligand_matches, lig_files)):
        for j, (fragment_match, fragment_file) in enumerate(zip(fragment_matches, fragment_files)):
            if lig_file and fragment_file:
                lig_sub = Chem.MolFromMolFile(lig_file, sanitize=True)
                frag_sub = Chem.MolFromMolFile(fragment_file, sanitize=True)
                sucosA = calc_sucos(frag_sub, lig_sub)
                sucosB = calc_sucos(lig_sub, frag_sub)
                sucos = (sucosA * sucosB) ** 0.5
                sucoses.append(sucos)

            else:
                sucoses.append(0)
            idxs.append((i, j))

    # get best matches with max sucos score
    best = sucoses.index(max(sucoses))
    best_ligand_match = ligand_matches[idxs[best][0]]
    best_frag_match = fragment_matches[idxs[best][1]]
    mapping = {lig_at: frag_at for lig_at, frag_at in zip(best_ligand_match, best_frag_match)}

    # using the matched atoms, retrieve the vector atoms in the ligand
    vectors = get_vector_atom(ligand, best_ligand_match)
    if len(vectors) == 0:
        print('no vectors')
        return None, None, None, None
    vectors_in_frag = [mapping[lig_at] for lig_at in vectors]
    # print(vectors)
    # print(vectors_in_frag)

    set_og_idxs(fragment, 'og_idx')
    fragment_with_hs = Chem.AddHs(fragment)
    new_vectors = [get_new_idx(fragment_with_hs, idx, 'og_idx') for idx in vectors_in_frag]
    new_mcs_atoms = [get_new_idx(fragment_with_hs, idx, 'og_idx') for idx in best_frag_match]

    # check vectors in frag for whether they are bonded to nothing else or ..
    passing_vectors = []
    vectors_are_hyd = []

    for vector in new_vectors:
        #print(vector)
        # print(vector)
        is_hyd, passing_vect = check_if_hydrogen_or_non_hydrogen(fragment_with_hs, vector, new_mcs_atoms)
        if passing_vect:
            vectors_are_hyd.append(is_hyd)
            passing_vectors.append(vector)

    shutil.rmtree(tmp_dir)
    return passing_vectors, vectors_are_hyd, mcs, max(sucoses)


def check_if_hydrogen_or_non_hydrogen(mol, vector_atom, mcs_atoms):
    """
    Check if the vector is A, one that using our protocol we recognise as a vector (either a hydrogen atom or terminal
    atom) and B, if so, which one of those it is


    :param mol:
    :param vector_atom:
    :param mcs_atoms:
    :return: (bool, bool), first indicating whether a hydrogen or terminal vector, second indicating whether it is recognised
    """
    # retrieve vector atom
    atom = mol.GetAtomWithIdx(vector_atom)

    # get its neighbours (not in MCS)
    h_neighbours = [neigh for neigh in atom.GetNeighbors() if neigh.GetAtomicNum() == 1 and neigh.GetIdx() not in mcs_atoms]
    neighbours = [neigh for neigh in atom.GetNeighbors() if neigh.GetAtomicNum() != 1 and neigh.GetIdx() not in mcs_atoms]

    # if it has no non-hydrogen neighbours (not in MCS), it is acceptable and a hydrogen vector
    if len(neighbours) == 0:
        if len(h_neighbours) > 0:
            return True, True  # whether hydrogen, whether valid
        else:
            return False, False

    # if it does have neighbours (not in MCS)
    if len(neighbours) > 0:

        # for each neighbour, check if it itself has OTHER neighbours (so whether we recognise it as a vector or not)
        neighbour_check = False
        for neighbour in neighbours:
            # get ITS neighbours that aren't the original vector atom
            neighbour_neighbours = [neighbour_neighbour for neighbour_neighbour in neighbour.GetNeighbors() if
                                neighbour_neighbour.GetAtomicNum() != 1 and neighbour_neighbour.GetIdx() != vector_atom]
            if len(neighbour_neighbours) == 0:
                neighbour_check = True
                break

        if neighbour_check:
            return False, True

        else:
            return False, False


def get_vector_atom(mol, mcs_matches):
    from elaboratability.utils.processUtils import check_vector_leads_to_elab_of_size
    vectors = set()
    for at_id in mcs_matches:
        atom = mol.GetAtomWithIdx(at_id)
        neighbours = [neigh.GetIdx() for neigh in atom.GetNeighbors()]
        for neigh in neighbours:
            if neigh not in mcs_matches:
                vectors.add(at_id)

    # check if vector leads to certain amount of atoms
    # TODO: check this - written very quickly
    filt_vectors = []
    for vector in vectors:
        vector_check = check_vector_leads_to_elab_of_size(mol, mcs_matches, vector)
        if vector_check:
            filt_vectors.append(vector)

    return filt_vectors


# import os
#
# test_dir = '/home/swills/F2L/Malhotra/2024-03-01_malhotra/malhotra_pairs/1F0T_PR1_3RXF_4AP'
# fragment_sdf = os.path.join(test_dir, '1F0T_PR1_3RXF_4AP_smaller_ligand.sdf')
# ligand_sdf = os.path.join(test_dir, '1F0T_PR1_3RXF_4AP_larger_ligand.sdf')
# get_vector_from_test_data(fragment_sdf, ligand_sdf)

