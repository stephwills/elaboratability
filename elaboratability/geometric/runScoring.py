
import os
from argparse import ArgumentParser
import pickle

from elaboratability.geometric.anchorScorer import AnchorScorer
from elaboratability.preprocessing.cloud import ClusterCloud
from elaboratability.utils.utils import dump_json
from rdkit import Chem
from tqdm import tqdm
import time
from joblib import Parallel, delayed


def eval_molecule(lig_name, name, output_dir, precursor, pdb_file, cloud, aizynth=False):
    """

    :param name:
    :param output_dir:
    :param precursor:
    :param pdb_file:
    :param cloud:
    :return:
    """
    lig_dir = os.path.join(output_dir, lig_name)
    if not os.path.exists(lig_dir):
        os.mkdir(lig_dir)

    results_dir = os.path.join(lig_dir, name)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    results_file = os.path.join(results_dir, f"{name}_results.pkl")
    scoring_file = os.path.join(results_dir, f"{name}_scored.pkl")

    if os.path.exists(results_file):
        return None
    start = time.time()

    try:
        if aizynth:
            from elaboratability.react.reactAnchorScorer import ReactionAnchorScorer
            scorer = ReactionAnchorScorer(precursor, pdb_file, cloud)
            keys = ['clashing_conf_ids', 'clashing_mol_ids', 'pdon_interactions', 'ldon_interactions', 'interacting_prot_ids', 'interacting_conf_ids', 'interacting_mol_ids', 'reacting_conf_ids', 'reacting_mol_ids', 'reacting_prot_ids']
        else:
            scorer = AnchorScorer(precursor, pdb_file, cloud)
            keys = ['clashing_conf_ids', 'clashing_mol_ids', 'pdon_interactions', 'ldon_interactions', 'interacting_prot_ids', 'interacting_conf_ids', 'interacting_mol_ids']
        scorer.prepare_molecules()
        scorer.get_vectors()
        scorer.evaluate_all_vectors()
        scorer.binary_scorer()

        with open(results_file, 'wb') as handle:
            results = scorer.results.copy()
            for vect in scorer.results:
                for k in scorer.results[vect]:
                    if k in keys:
                        results[vect][k] = len(scorer.results[vect][k])
                    else:
                        results[vect][k] = scorer.results[vect][k]
            pickle.dump(results, handle)

        with open(scoring_file, 'wb') as handle:
            pickle.dump(scorer.scored, handle)
    except Exception as e:
        print('Error for', name, e)
        dump_json(str(e), os.path.join(results_dir, 'error.json'))

    end = time.time()
    time_taken = end-start
    dump_json(time_taken, os.path.join(results_dir, 'time_taken.json'))


def main():
    """

    :return:
    """
    parser = ArgumentParser()
    parser.add_argument('--precursor_dir')
    parser.add_argument('--pdbbind_dir')
    parser.add_argument('--clustered_conformer_file')
    parser.add_argument('--clustered_conformer_json')
    parser.add_argument('--processed_data_file')
    parser.add_argument('--cloud_file')
    parser.add_argument('--output_dir')
    parser.add_argument('--n_cpus', type=int)
    parser.add_argument('--aizynth', action='store_true')
    args = parser.parse_args()

    start = time.time()

    print('Initializing cloud')
    cloud = ClusterCloud(args.clustered_conformer_file,
                         args.clustered_conformer_json,
                         args.processed_data_file,
                         False,
                         args.cloud_file,
                         False)
    cloud.process_cloud()

    prec_files = [i for i in os.listdir(args.precursor_dir) if 'sdf' in i]
    prec_names = [i.replace('.sdf', '') for i in prec_files]
    prec_files = [os.path.join(args.precursor_dir, i) for i in prec_files]
    precursors = [Chem.SDMolSupplier(i)[0] for i in prec_files]
    lig_names = [i.split('-')[0] for i in prec_names]
    pdb_files = [os.path.join(args.pdbbind_dir, lig_code, f"{lig_code}_protein_cleaned.pdb") for lig_code in lig_names] #code_protein_cleaned.pdb
    print(len(lig_names), 'precursors loaded')

    print('Beginning scoring')
    Parallel(n_jobs=args.n_cpus, backend='multiprocessing')(
        delayed(eval_molecule)(lig_name, prec_name, args.output_dir, prec, pdb_file, cloud, aizynth=args.aizynth)
        for lig_name, prec_name, prec, pdb_file in tqdm(zip(lig_names, prec_names, precursors, pdb_files), total=len(lig_names), position=0, leave=True)
    )

    end = time.time()
    time_taken = end-start

    timings_file = os.path.join(args.output_dir, 'timings')
    d = {'time': time_taken,
         'n_cpus': args.n_cpus}
    dump_json(d, timings_file)


if __name__ == "__main__":
    main()
