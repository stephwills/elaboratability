
import os
import pickle
from argparse import ArgumentParser

from elaboratability.preprocessing.labelValidationVectors import Labeller
from elaboratability.utils.utils import load_json
from rdkit import Chem
from tqdm import tqdm


def run_labelling(ligand, pdb, smi, decorators, aizynth_check, num_seeds, sucos_threshold, overlap_threshold, n_placed_threshold, n_cpus, output_file):
    """

    :param ligand:
    :param pdb:
    :param smi:
    :param decorators:
    :param aizynth_check:
    :param num_seeds:
    :param sucos_threshold:
    :param overlap_threshold:
    :param n_placed_threshold:
    :param n_cpus:
    :param output_file:
    :return:
    """
    labeller = Labeller(smi, ligand, pdb, decorators, aizynth_check, num_seeds, sucos_threshold, overlap_threshold,
                        n_placed_threshold, n_cpus)
    labeller.label_vectors()

    with open(output_file, 'wb') as handle:
        pickle.dump(labeller.labelled_vectors, handle)


def main():
    """

    :return:
    """
    parser = ArgumentParser()
    parser.add_argument('--pdbbind_dir')
    parser.add_argument('--decs_json')
    parser.add_argument('--output_dir')
    parser.add_argument('--aizynth_check', action='store_true')
    parser.add_argument('--num_seeds', type=int)
    parser.add_argument('--sucos_threshold', type=float)
    parser.add_argument('--overlap_threshold', type=float)
    parser.add_argument('--n_placed_threshold', type=int)
    parser.add_argument('--n_cpus', type=int)
    args = parser.parse_args()

    decorators = load_json(args.decs_json)

    input_dir = args.pdbbind_dir
    sub_dirs = [dir for dir in os.listdir(input_dir) if 'zip' not in dir]
    print(len(sub_dirs), 'dirs in directory')
    sub_dirs.sort()

    sdfs = [os.path.join(input_dir, lig, f"{lig}_ligand.sdf") for lig in sub_dirs]
    pdbs_unfiltered = [os.path.join(input_dir, lig, f"{lig}_protein_cleaned.pdb") for lig in sub_dirs]

    mols, smiles, pdbs, lig_codes = [], [], [], []

    print('Loading mols and smiles')
    for sdf, pdb, lig_code in tqdm(zip(sdfs, pdbs_unfiltered, sub_dirs), total=len(sdfs), position=0, leave=True):
        if os.path.exists(sdf) and os.path.exists(pdb):
            mol = Chem.SDMolSupplier(sdf)[0]
            if mol:
                smi = Chem.MolToSmiles(mol)
                mols.append(mol)
                smiles.append(smi)
                pdbs.append(pdb)
                lig_codes.append(lig_code)
    print(len(mols), 'mols for evaluating')

    output_files = [os.path.join(args.output_dir, f"{lig_code}.pkl") for lig_code in lig_codes]

    # TODO: where to parallelise?
    # from joblib import Parallel, delayed
    # Parallel(n_jobs=args.n_cpus, backend="multiprocessing")(
    #     delayed(run_labelling)(
    #         ligand, pdb, smi, decorators, args.aizynth_check, args.num_seeds, args.sucos_threshold, args.overlap_threshold, args.n_placed_threshold, args.n_cpus, output_file
    #     ) for ligand, pdb, smi, output_file in tqdm(zip(mols, pdbs, smiles, output_files), total=len(mols), position=0, leave=True)
    # )

    for ligand, pdb, smi, output_file in tqdm(zip(mols, pdbs, smiles, output_files), total=len(mols), position=0, leave=True):
        run_labelling(ligand, pdb, smi, decorators, args.aizynth_check, args.num_seeds, args.sucos_threshold, args.overlap_threshold, args.n_placed_threshold, args.n_cpus, output_file)


if __name__ == "__main__":
    main()
