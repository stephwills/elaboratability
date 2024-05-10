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
    if verbose: print('Attempting placement for precursor:', precursor_smi)
    # get possible atom mappings between the precursor and the molecule
    maps, mcs = get_mappings_between_precursor_and_mol(mol, precursor)

    # check there is a sensible number of shared atoms between precursor and molecule
    if mcs.GetNumAtoms() < min_mcs_atoms or precursor.GetNumAtoms() - mcs.GetNumAtoms() > max_extra_precursor_atoms:
        return False, False

    placed_precursors = []
    property_dicts = []

    # for each map, generate placed precursors
    for map in maps:
        vectors, vectors_molidx, vector_neighbours = get_vector_atoms(mol, precursor, map)
        if verbose: print('Vectors in mol:', vectors_molidx)
        passing_vectors = []
        passing_vector_neighbours = []

        # check if the elaboration from vector makes an interaction with the protein and
        # check if an atom in the elaboration is closer than an atom in the mcs
        for vector, vector_mol, neighbours in zip(vectors, vectors_molidx, vector_neighbours):

            check = check_elaboration_is_useful(mol, vector_mol, map, int_idxs, atom_dists)
            if check:
                print(vector, 'vector passes')
                passing_vectors.append(vector)
                passing_vector_neighbours.append(neighbours)
            else:
                print(vector, 'vector fails')

        placed_precursor = constrained_embed_of_precursor(mol, precursor, mcs, map)
        if placed_precursor and len(passing_vectors) > 0:
            properties = {}
            # record properties in dict that can be added as mol props later
            passing_vectors = ",".join([str(i) for i in passing_vectors])
            properties['vectors'] = passing_vectors
            passing_vector_neighbours_string = ['-'.join([str(i) for i in l]) for l in passing_vector_neighbours]
            passing_vector_neighbours_string = ','.join(passing_vector_neighbours_string)
            properties['vector_neighbours'] = passing_vector_neighbours_string
            mol_substruct_match_string = ','.join([str(i) for i in map.keys()])
            precursor_substruct_match_string = ','.join([str(i) for i in map.values()])
            properties['mol_substruct_match'] = mol_substruct_match_string
            properties['precursor_substruct_match'] = precursor_substruct_match_string
            properties['smiles'] = precursor_smi
            placed_precursors.append(placed_precursor)
            property_dicts.append(properties)

    return placed_precursors, property_dicts


def retrieve_precursors_for_mol(pdb_file: str, output_file: str, sdf_file = None, mol = None):
    """

    :param mol:
    :param smiles:
    :param pdb_file:
    :param output_file:
    :return:
    """
    if sdf_file:
        mol = Chem.SDMolSupplier(sdf_file)[0]
    smiles = Chem.MolToSmiles(mol)

    # retrieve possible precursors predicted by aizynth
    precursor_smiles = get_reactants_using_aizynth(smiles)

    # for each ligand atom, get the closest distance to a protein atom AND
    # get the ligand atom ids that are within distance of an hbonder in protein
    interacting_ids, closest_dists = get_possible_interacting_atoms_and_min_dists(mol, pdb_file)

    all_precursors = []
    all_properties = []

    for precursor_smi in precursor_smiles:
        placed_precursors, properties = place_precursor(mol, precursor_smi, interacting_ids, closest_dists)
        if placed_precursors:
            all_precursors.extend(placed_precursors)
            all_properties.extend(properties)

    if len(all_precursors) > 0:
        w = Chem.SDWriter(output_file)
        for precursor, properties in zip(all_precursors, all_properties):
            add_properties_from_dict(precursor, properties)
            w.write(precursor)
        w.close()


def parallel_precursor_enumeration(sdf_files: list, pdb_files: list, output_files: list, n_cpus: int):
    """

    :param sdf_files:
    :param pdb_files:
    :param output_files:
    :param n_cpus:
    :return:
    """
    #pdb_file: str, output_file: str, sdf_file = None, mol = None
    Parallel(n_jobs=n_cpus, backend="multiprocessing")(
        delayed(retrieve_precursors_for_mol)(pdb_file, output_file, sdf_file)
        for sdf_file, pdb_file, output_file in tqdm(zip(sdf_files, pdb_files, output_files), position=0, leave=True, total=len(sdf_files))
    )


def main():
    parser = ArgumentParser()
    parser.add_argument('--txt_file', help='each line is comma delimited ligand_name, sdf file and pdb file')
    parser.add_argument('--output_dir')
    parser.add_argument('--n_cpus', type=int)
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    with open(args.txt_file, "r+") as f:
        files = [x.strip() for x in f.readlines()]
    ligands = [file.split(',')[0] for file in files]
    sdf_files = [file.split(',')[1] for file in files]
    pdb_files = [file.split(',')[2] for file in files]
    output_files = [os.path.join(args.output_dir, f"{ligand}_precursors.sdf") for ligand in ligands]

    start = time.time()

    print(len(ligands), 'mols read')
    if args.parallel:
        parallel_precursor_enumeration(sdf_files, pdb_files, output_files, args.n_cpus)

    else:
        for sdf_file, pdb_file, output_file in tqdm(zip(sdf_files, pdb_files, output_files), position=0, leave=True, total=len(sdf_files)):
            retrieve_precursors_for_mol(pdb_file, output_file, sdf_file)

    end = time.time()
    time_taken = round(end-start, 2)
    d = {'n_mols': len(ligands),
         'time': time_taken}
    dump_json(d, os.path.join(args.output_dir, 'time_taken.json'))


if __name__ == "__main__":
    main()
