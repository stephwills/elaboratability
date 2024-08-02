
import os
from argparse import ArgumentParser

from elaboratability.utils.processUtils import (
    get_mappings_between_precursor_and_mol, get_vector_atoms)
from elaboratability.utils.utils import add_properties_from_dict
from rdkit import Chem
from joblib import Parallel, delayed
from tqdm import tqdm


def reprocess_precursor(precursor_name, precursor_file, pdbbind_dir, output_dir, min_mcs_atoms=3, max_extra_precursor_atoms=6):
    lig_code = precursor_name.split('-')[0]
    mol_sdf = os.path.join(pdbbind_dir, lig_code, f"{lig_code}_ligand.sdf")
    pdb_file = os.path.join(pdbbind_dir, lig_code, f"{lig_code}_protein_cleaned.pdb")
    mol = Chem.SDMolSupplier(mol_sdf)[0]

    precursor_mol = Chem.SDMolSupplier(precursor_file)[0]
    precursor_smi = precursor_mol.GetProp('smiles')
    precursor = Chem.MolFromSmiles(precursor_smi)

    # get possible atom mappings between the precursor and the molecule
    maps, mcs = get_mappings_between_precursor_and_mol(mol, precursor)
    if len(maps) == 0:
        return None

    # check there is a sensible number of shared atoms between precursor and molecule
    if mcs.GetNumAtoms() < min_mcs_atoms or precursor.GetNumAtoms() - mcs.GetNumAtoms() > max_extra_precursor_atoms:
        return None

    mol_substruct_match = precursor_mol.GetProp('mol_substruct_match').split(',')
    precursor_substruct_match = precursor_mol.GetProp('precursor_substruct_match').split(',')
    map = {int(mol_at): int(prec_at) for mol_at, prec_at in zip(mol_substruct_match, precursor_substruct_match)}

    vectors, vectors_molidx, vector_neighbours, vector_types = get_vector_atoms(mol, precursor, map)
    passing_vectors = []
    passing_vector_types = []

    # check if the elaboration from vector makes an interaction with the protein and
    # check if an atom in the elaboration is closer than an atom in the mcs
    for vector, vector_mol, neighbours, vector_type in zip(vectors, vectors_molidx, vector_neighbours, vector_types):
        if vector_type == 'hydrogen':
            passing_vectors.append(vector)
            passing_vector_types.append(vector_type)
        else:
            for neighbour in neighbours:
                passing_vectors.append(neighbour)
                passing_vector_types.append(vector_type)

    new_file = os.path.join(output_dir, f"{precursor_name}.sdf")

    if len(passing_vectors) > 0:
        new_mol = Chem.Mol(precursor_mol)
        new_mol.ClearProp('vector_neighbours')
        properties = {}
        # record properties in dict that can be added as mol props later
        passing_vectors = ",".join([str(i) for i in passing_vectors])
        properties['vectors'] = passing_vectors
        mol_substruct_match_string = ','.join([str(i) for i in map.keys()])
        precursor_substruct_match_string = ','.join([str(i) for i in map.values()])
        properties['mol_substruct_match'] = mol_substruct_match_string
        properties['precursor_substruct_match'] = precursor_substruct_match_string
        properties['smiles'] = precursor_smi
        properties['vector_types'] = ",".join([str(i) for i in passing_vector_types])
        add_properties_from_dict(new_mol, properties)
        w = Chem.SDWriter(new_file)
        w.write(new_mol)
        w.close()
        print(new_file, 'written')
    else:
        print(new_file, 'not writtten')


def main():
    parser = ArgumentParser()
    parser.add_argument('--precursor_dir')
    parser.add_argument('--pdbbind_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--n_cpus', type=int)
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    precursor_files = [i for i in os.listdir(args.precursor_dir) if 'zip' not in i]
    precursors = [i.replace('.sdf', '') for i in precursor_files]
    precursor_files = [os.path.join(args.precursor_dir, file) for file in precursor_files]

    Parallel(n_jobs=args.n_cpus, backend="multiprocessing")(
        delayed(reprocess_precursor)(precursor_name, precursor_file, args.pdbbind_dir, args.output_dir)
        for precursor_name, precursor_file in tqdm(zip(precursors, precursor_files), position=0, leave=True, total=len(precursors))
    )


if __name__ == "__main__":
    main()
