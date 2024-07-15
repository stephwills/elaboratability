"""
Process docking data into data that can be used to train the elaboratability model
"""

import os
from argparse import ArgumentParser
import time

import elaboratability.utils.processConfig as config
from aizynthfinder.aizynthfinder import AiZynthFinder
from elaboratability.utils.processUtils import (
    check_elaboration_is_useful, constrained_embed_of_precursor,
    get_mappings_between_precursor_and_mol,
    get_possible_interacting_atoms_and_min_dists, get_vector_atoms)
from elaboratability.utils.utils import add_properties_from_dict, dump_json
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import Mol
from tqdm import tqdm


def get_reactants_using_aizynth(smi: str, aizynth_config_file=config.AIZYNTH_CONFIG_FILE, stock=config.STOCK,
                                expansion_policy=config.EXPANSION_POLICY, filter_policy=config.FILTER_POLICY,
                                verbose=config.VERBOSE) -> list:
    """
    Retrieve possible reactants for a molecule using AiZynthFinder

    :param smi:
    :param aizynth_config_file:
    :param stock:
    :param expansion_policy:
    :param filter_policy:
    :return:
    """
    finder = AiZynthFinder(configfile=aizynth_config_file)
    finder.stock.select(stock)
    finder.expansion_policy.select(expansion_policy)
    finder.filter_policy.select(filter_policy)
    finder.target_smiles = smi
    finder.tree_search()
    finder.build_routes()
    stats = finder.extract_statistics()
    precursors = stats['precursors_in_stock'].split(', ')
    assert len(precursors) > 0, f"No precursors for smi {smi}"
    if verbose: print('Precursors from AiZynth finder:', precursors)
    return precursors


def place_precursor(mol: Mol, precursor_smi: str, int_idxs: list, atom_dists: dict, min_mcs_atoms=3,
                    max_extra_precursor_atoms=6, verbose=config.VERBOSE):
    """

    :param mol:
    :param precursor_smi:
    :param int_idxs:
    :param atom_dists:
    :param min_mcs_atoms:
    :param max_extra_precursor_atoms:
    :param verbose:
    :return:
    """
    precursor = Chem.MolFromSmiles(precursor_smi)
    if not precursor:
        return None, None
    if verbose: print('Attempting placement for precursor:', precursor_smi)
    # get possible atom mappings between the precursor and the molecule
    maps, mcs = get_mappings_between_precursor_and_mol(mol, precursor)
    if len(maps) == 0:
        return None, None

    # check there is a sensible number of shared atoms between precursor and molecule
    if mcs.GetNumAtoms() < min_mcs_atoms or precursor.GetNumAtoms() - mcs.GetNumAtoms() > max_extra_precursor_atoms:
        return None, None

    placed_precursors = []
    property_dicts = []

    # for each map, generate placed precursors
    for map in maps:
        vectors, vectors_molidx, vector_neighbours, vector_types = get_vector_atoms(mol, precursor, map)
        if verbose: print('Vectors in mol:', vectors_molidx)
        passing_vectors = []
        passing_vector_neighbours = []
        passing_vector_types = []

        # check if the elaboration from vector makes an interaction with the protein and
        # check if an atom in the elaboration is closer than an atom in the mcs
        for vector, vector_mol, neighbours, vector_type in zip(vectors, vectors_molidx, vector_neighbours, vector_types):

            check = check_elaboration_is_useful(mol, vector_mol, map, int_idxs, atom_dists)
            if check:
                # print(vector, 'vector passes')
                if vector_type == 'hydrogen':
                    passing_vectors.append(vector)
                    passing_vector_types.append(vector_type)
                else:
                    for neighbour in neighbours:
                        passing_vectors.append(neighbour)
                        passing_vector_types.append(vector_type)
               #passing_vector_neighbours.append(neighbours)
               #passing_vector_types.append(vector_type)
            # else:
                # print(vector, 'vector fails')

        placed_precursor = constrained_embed_of_precursor(mol, precursor, mcs, map)
        if placed_precursor and len(passing_vectors) > 0:
            properties = {}
            # record properties in dict that can be added as mol props later
            passing_vectors = ",".join([str(i) for i in passing_vectors])
            properties['vectors'] = passing_vectors
            #passing_vector_neighbours_string = ['-'.join([str(i) for i in l]) for l in passing_vector_neighbours]
            #passing_vector_neighbours_string = ','.join(passing_vector_neighbours_string)
            #properties['vector_neighbours'] = passing_vector_neighbours_string
            mol_substruct_match_string = ','.join([str(i) for i in map.keys()])
            precursor_substruct_match_string = ','.join([str(i) for i in map.values()])
            properties['mol_substruct_match'] = mol_substruct_match_string
            properties['precursor_substruct_match'] = precursor_substruct_match_string
            properties['smiles'] = precursor_smi
            properties['vector_types'] = ",".join([str(i) for i in passing_vector_types])
            placed_precursors.append(placed_precursor)
            property_dicts.append(properties)

    return placed_precursors, property_dicts


def retrieve_precursors_for_mol(pdb_file: str, sdf_file = None, lig_code: str = None, output_dir: str = None, mol = None):
    """

    :param mol:
    :param smiles:
    :param pdb_file:
    :param output_file:
    :return:
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(pdb_file):
        print('Input pdb file does not exist')
        return None
    if sdf_file:
        if not os.path.exists(sdf_file):
            print('Input sdf file does not exist')
            return None

    try:
        if sdf_file:
            mol = Chem.SDMolSupplier(sdf_file)[0]
        if not mol:
            return None
        smiles = Chem.MolToSmiles(mol)

        # retrieve possible precursors predicted by aizynth
        precursor_smiles = get_reactants_using_aizynth(smiles)

        if len(precursor_smiles) == 0:
            return None

        # for each ligand atom, get the closest distance to a protein atom AND
        # get the ligand atom ids that are within distance of an hbonder in protein
        interacting_ids, closest_dists = get_possible_interacting_atoms_and_min_dists(mol, pdb_file)

        all_precursors = []
        all_properties = []

        for precursor_smi in precursor_smiles:
            placed_precursors, properties = place_precursor(mol, precursor_smi, interacting_ids, closest_dists)
            if placed_precursors:
                if len(placed_precursors) > 0:
                    all_precursors.extend(placed_precursors)
                    all_properties.extend(properties)

        if len(all_precursors) > 0:

            for i, (precursor, properties) in enumerate(zip(all_precursors, all_properties)):
                output_file = os.path.join(output_dir, f"{lig_code}_{i}.sdf")
                w = Chem.SDWriter(output_file)
                add_properties_from_dict(precursor, properties)
                w.write(precursor)
                w.close()
    except Exception as e:
        print(e)
        print('Failed for', lig_code)


def parallel_precursor_enumeration(sdf_files: list, pdb_files: list, lig_codes: list, output_dir: str, n_cpus: int):
    """

    :param sdf_files:
    :param pdb_files:
    :param output_files:
    :param n_cpus:
    :return:
    """
    Parallel(n_jobs=n_cpus, backend="multiprocessing")(
        delayed(retrieve_precursors_for_mol)(pdb_file, sdf_file, lig_code, output_dir)
        for sdf_file, pdb_file, lig_code in tqdm(zip(sdf_files, pdb_files, lig_codes), position=0, leave=True, total=len(sdf_files))
    )


def main():
    parser = ArgumentParser()
    parser.add_argument('--pdbbind_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--n_cpus', type=int)
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    dir = args.pdbbind_dir
    lig_codes = [i for i in os.listdir(dir) if 'zip' not in i]
    sdf_files = [os.path.join(dir, lig_code, f"{lig_code}_ligand.sdf") for lig_code in lig_codes]
    pdb_files = [os.path.join(dir, lig_code, f"{lig_code}_protein_cleaned.pdb") for lig_code in lig_codes]

    start = time.time()

    print(len(lig_codes), 'mols read')
    if args.parallel:
        parallel_precursor_enumeration(sdf_files, pdb_files, lig_codes, args.output_dir, args.n_cpus)

    else:
        for sdf_file, pdb_file, lig_code in tqdm(zip(sdf_files, pdb_files, lig_codes), position=0, leave=True, total=len(sdf_files)):
            retrieve_precursors_for_mol(pdb_file, sdf_file, lig_code, args.output_dir)

    end = time.time()
    time_taken = round(end-start, 2)
    d = {'n_mols': len(lig_codes),
         'time': time_taken}
    dump_json(d, os.path.join(args.output_dir, 'time_taken.json'))


if __name__ == "__main__":
    main()
