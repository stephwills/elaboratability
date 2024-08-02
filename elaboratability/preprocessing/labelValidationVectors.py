
import logging
import os
import shutil
import tempfile

from elaboratability.react.createReaction import (
    create_reaction_smiles_for_decorators,
    filter_multiple_rxns_for_vector_with_aizynthfinder)
from elaboratability.utils.geometricUtils import (
    get_hydrogen_vector_pairs, get_non_hydrogen_vector_pairs)
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import rdShapeHelpers
from tqdm import tqdm


class Labeller():

    def __init__(self, smiles, ligand, pdb, decorators, aizynth_check=False, num_seeds=1, sucos_threshold=0.55, overlap_threshold=0.3,
                 n_placed_threshold=5, n_cpus=1, verbose=False):
        """

        :param smiles:
        :param ligand:
        :param pdb:
        :param decorators:
        :param aizynth_check:
        :param num_seeds:
        :param sucos_threshold:
        :param overlap_threshold:
        :param n_placed_threshold:
        :param n_cpus:
        :param verbose:
        """
        self.smiles = smiles
        self.ligand = Chem.AddHs(ligand, addCoords=True)
        self.protein = Chem.MolFromPDBFile(pdb)
        self.pdb = pdb
        self.decorators = decorators
        self.aizynth_check = aizynth_check
        self.num_seeds = num_seeds
        self.sucos_threshold = sucos_threshold
        self.overlap_threshold = overlap_threshold
        self.n_placed_threshold = n_placed_threshold
        self.n_cpus = n_cpus
        self.verbose = verbose

        # initialize vectors
        self.get_vectors()

    def get_vectors(self):
        """

        :return:
        """
        unfilt_h_pairs = get_hydrogen_vector_pairs(self.ligand, False)
        h_pairs = []

        # remove duplicate vectors (e.g. same orig atom, both attached to a hydrogen)
        h_orig_atoms = []
        for pair in unfilt_h_pairs:
            if pair[0] not in h_orig_atoms:
                h_orig_atoms.append(pair[0])
                h_pairs.append(pair)

        non_h_pairs = get_non_hydrogen_vector_pairs(self.ligand)
        self.vector_is_hyd = {}
        for h_pair in h_pairs:
            self.vector_is_hyd[tuple(h_pair)] = True
        for non_h_pair in non_h_pairs:
            self.vector_is_hyd[tuple(non_h_pair)] = False
        self.vector_pairs = h_pairs + non_h_pairs

    def label_vectors(self):
        """

        :return:
        """
        self.labelled_vectors = {}
        for i, pair in enumerate(self.vector_pairs):
            if self.verbose: print('Evaluating pair', f'{i+1}/{len(self.vector_pairs)}: {pair}')
            products = self.generate_elaborations(pair)
            if self.verbose: print(len(products), 'products')
            product_results = Parallel(backend='multiprocessing', n_jobs=self.n_cpus)(
                delayed(self.wictor_placement_and_overlap_check)(product)
                for product in tqdm(products, total=len(products), position=0, leave=True)
            )
            n_prods = sum(product_results)
            if self.verbose: print(n_prods, 'products pass placement check')
            if n_prods >= self.n_placed_threshold:
                res = True
            else:
                res = False
            self.labelled_vectors[tuple(pair)] = res
            if self.verbose: print('Result:', res)

    def generate_elaborations(self, pair):
        """

        :param pair:
        :return:
        """
        if self.aizynth_check:
            filter_res, _, _, products = filter_multiple_rxns_for_vector_with_aizynthfinder(
                self.ligand, pair[0], pair[1], self.vector_is_hyd[tuple(pair)],
                self.decorators, return_reaction_smiles=True
            )
            products = [prod for prod, res in zip(products, filter_res) if res]

        else:
            _, products = create_reaction_smiles_for_decorators(
                self.ligand, self.decorators, pair[0], pair[1], self.vector_is_hyd[tuple(pair)]
            )

        return products

    def wictor_placement_and_overlap_check(self, product):
        """

        :param self:
        :param product:
        :return:
        """
        from elaboratability.preprocessing.prepareCrystalData import calc_sucos
        from fragmenstein import Wictor
        good_overlap = 0

        for i in range(self.num_seeds):
            tmp_dir = tempfile.mkdtemp(prefix='/tmp/')
            wictor = Wictor(hits=[self.ligand], pdb_filename=self.pdb, monster_random_seed=i)
            wictor.work_path = tmp_dir
            wictor.place(product, long_name=f'{os.path.basename(tmp_dir)}_{i}')
            wictor.enable_stdout(level=logging.ERROR)

            placed = wictor.minimized_mol
            suc = calc_sucos(self.ligand, placed)
            if suc >= self.sucos_threshold:
                overlap = 1 - rdShapeHelpers.ShapeProtrudeDist(placed, self.protein, allowReordering=True)
                if overlap <= self.overlap_threshold:
                    good_overlap += 1
            shutil.rmtree(tmp_dir)

        if good_overlap > 0:
            return True
        else:
            return False
