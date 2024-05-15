
import elaboratability.utils.processConfig as proConfig
from elaboratability.utils.geometricUtils import *
from elaboratability.utils.pymolUtils import save_close_protein_atoms
from elaboratability.utils.utils import get_intersect
from rdkit import Chem
from scipy.spatial.distance import cdist


class AnchorScorer():

    def __init__(self, ligand, pdb_file, cloud, pocket_dist=config.POCKET_DIST, clash_cutoff=config.CLASH_CUTOFF,
                 hbond_cutoff=config.HBOND_CUTOFF, cloud_attach_coords=config.CLOUD_ATTACH_COORDS,
                 cloud_adj_coords=config.CLOUD_ADJ_COORDS, atom_dict=proConfig.ATOM_DICT, check_lig_clash=config.CHECK_LIG_CLASH,
                 check_for_ints=config.CHECK_FOR_INTS, total_mol_elabs=config.TOTAL_MOLS, total_confs=config.TOTAL_CONFS, min_prop_mols_added=config.MIN_PROP_MOLS_ADDED,
                 max_prop_mols_added=config.MAX_PROP_MOLS_ADDED, min_ints_reached=config.MIN_INTS_REACHED, verbose=False):
        """

        :param ligand:
        :param pdb_file:
        :param cloud:
        :param pocket_dist:
        :param clash_cutoff:
        :param hbond_cutoff:
        :param cloud_attach_coords:
        :param cloud_adj_coords:
        :param atom_dict:
        :param check_lig_clash:
        :param check_for_ints:
        :param total_mol_elabs:
        :param total_confs:
        :param min_prop_mols_added:
        :param max_prop_mols_added:
        :param min_ints_reached:
        :param verbose:
        """
        self.ligand = Chem.AddHs(ligand, addCoords=True)
        self.pdb_file = pdb_file
        self.cloud = cloud

        self.atom_dict = atom_dict
        self.cloud_radii = [self.atom_dict[point.elem]['atomic_radii'] for point in self.cloud.cluster_points]

        # scorer settings
        self.check_lig_clash = check_lig_clash
        self.check_for_ints = check_for_ints
        self.total_mol_elabs = total_mol_elabs
        self.total_confs = total_confs
        self.min_prop_mols_added = min_prop_mols_added
        self.max_prop_mols_added = max_prop_mols_added
        self.min_mols_added = self.total_mol_elabs * self.min_prop_mols_added
        self.max_mols_added = self.total_mol_elabs * self.max_prop_mols_added
        self.min_ints_reached = min_ints_reached
        self.verbose = verbose

        # thresholds
        self.pocket_dist = pocket_dist
        self.clash_cutoff = clash_cutoff
        self.hbond_cutoff = hbond_cutoff

        # for alignment
        self.cloud_attach_coords = cloud_attach_coords
        self.cloud_adj_coords = cloud_adj_coords

        self.results = {}

    def prepare_molecules(self):
        """

        :return:
        """
        self.ligand_data = get_ligand_data(self.ligand)
        all_ligand_coords = [self.ligand_data[atom]['coords'] for atom in range(self.ligand.GetNumAtoms())]
        self.pocket_data = save_close_protein_atoms(self.pdb_file, all_ligand_coords, self.pocket_dist)
        self.pocket_coords, self.pocket_ids, self.pocket_elems = np.array(self.pocket_data['coords']), self.pocket_data['IDs'], self.pocket_data['elems']
        self.pocket_radii = [self.atom_dict[elem]['atomic_radii'] for elem in self.pocket_elems]

        # get whether there are potential h bonders
        if self.check_for_ints:
            from elaboratability.utils.pymolUtils import (findHDonorsAcceptors,
                                                          findSurfaceAtoms)

            # process the protein pocket to find H donors and acceptors
            surface_ids = findSurfaceAtoms(self.pdb_file)['IDs']  # get the IDs of surface atoms from the full pdb file

            hbonder_data = findHDonorsAcceptors(self.pdb_file)
            hbond_donor_ids = hbonder_data["donor_IDs"]
            hbond_donor_ids = get_intersect(hbond_donor_ids, self.pocket_ids)  # select only those in pocket
            hbond_donor_ids = get_intersect(hbond_donor_ids, surface_ids)  # select only those on surface

            hbond_acceptor_ids = hbonder_data["acceptor_IDs"]
            hbond_acceptor_ids = get_intersect(hbond_acceptor_ids, self.pocket_ids)  # select only those in pocket
            hbond_acceptor_ids = get_intersect(hbond_acceptor_ids, surface_ids)  # select only those on surface

            # convert into a bool list
            self.prot_is_don = [id in hbond_donor_ids for id in self.pocket_ids]
            self.prot_is_acc = [id in hbond_acceptor_ids for id in self.pocket_ids]

    def get_vectors(self):
        """

        :return:
        """
        self.vectors = get_vectors_with_coords(self.ligand)

    def get_ligand_data_for_eval(self, anchor_atom, replaced_atom_idxs):
        """

        :param anchor_atom:
        :param replaced_atom:
        :return:
        """
        ligand_atoms = [atom for atom in self.ligand_data if atom != anchor_atom and atom not in replaced_atom_idxs \
                        and self.ligand.GetAtomWithIdx(atom).GetAtomicNum() != 1]
        ligand_coords = np.array([self.ligand_data[atom]['coords'] for atom in ligand_atoms])
        ligand_radii = [self.ligand_data[atom]['radii'] for atom in ligand_atoms]
        return ligand_atoms, ligand_coords, ligand_radii

    def align_coords(self, anchor_coords, replaced_coords):
        """

        :param anchor_atom:
        :param replaced_atom:
        :return:
        """
        # anchor_coords, replaced_coords = self.ligand_data[anchor_atom]['coords'], self.ligand_data[replaced_atom]['coords']
        aligned_coords = align_cloud_to_vector(replaced_coords, anchor_coords, self.cloud.coords, self.cloud_attach_coords, self.cloud_adj_coords)
        return np.array(aligned_coords)

    def check_for_clashes(self, dists, mol_radii):
        """

        :param dists:
        :param mol_radii:
        :return:
        """
        # assume dists are between mol (ligand or protein) and the cloud
        thresh = np.where(dists < self.clash_cutoff)

        mol_clash_idxs = []
        cloud_clash_idxs = []

        for mol_idx, cloud_idx in zip(thresh[0], thresh[1]):
            cutoff = 0.63 * (mol_radii[mol_idx] + self.cloud_radii[cloud_idx])
            if dists[mol_idx, cloud_idx] < cutoff:
                mol_clash_idxs.append(mol_idx)
                cloud_clash_idxs.append(cloud_idx)

        return mol_clash_idxs, cloud_clash_idxs

    def check_for_potential_interactions(self, dists, clash_conf_ids):
        """

        :param dists:
        :param clash_conf_ids:
        :return:
        """
        pdon_interactions = []
        ldon_interactions = []

        interacting_prot_ids = set()
        interacting_conf_ids = set()

        # self.prot_is_don, self.prot_is_acc, whether cluster point associated with don or acc
        thresh = np.where(dists < self.hbond_cutoff)
        for prot_idx, cloud_idx in zip(thresh[0], thresh[1]):
            # check for potential interactions where protein is donor
            if self.prot_is_don[prot_idx] and self.cloud.cluster_points[cloud_idx].acc_flag:
                # check the conformers are not clashing
                for conf in self.cloud.cluster_points[cloud_idx].conf_ids:
                    if conf not in clash_conf_ids:
                        pdon_interactions.append([prot_idx, cloud_idx, conf])
                        interacting_prot_ids.add(prot_idx)
                        interacting_conf_ids.add(conf)

            # check for potential interactions where protein is acceptor
            if self.prot_is_acc[prot_idx] and self.cloud.cluster_points[cloud_idx].don_flag:
                # check the conformers are not clashing
                for conf in self.cloud.cluster_points[cloud_idx].conf_ids:
                    if conf not in clash_conf_ids:
                        ldon_interactions.append([prot_idx, cloud_idx, conf])
                        interacting_prot_ids.add(prot_idx)
                        interacting_conf_ids.add(conf)

        return pdon_interactions, ldon_interactions, interacting_prot_ids, interacting_conf_ids

    def evaluate(self, vector):
        """

        :param vector: Vector object
        :return:
        """
        # get the coordinates of the other atoms in the ligand (not including the vector)
        ligand_atoms, ligand_coords, ligand_radii = self.get_ligand_data_for_eval(vector.anchor_atom_idx,
                                                                                  vector.replace_atom_idxs)

        # align the coordinates of the cloud to the vector
        aligned_coords = self.align_coords(vector.anchor_coord, vector.replace_coord)

        ############################## check for clash with the ligand ##############################
        cloud_lig_clash_idxs = []
        if self.check_lig_clash:
            lig_dists = cdist(ligand_coords, aligned_coords)
            lig_clash_idxs, cloud_lig_clash_idxs = self.check_for_clashes(lig_dists, ligand_radii)

        # check for clash with protein atoms
        prot_dists = cdist(self.pocket_coords, aligned_coords)
        prot_clash_idxs, cloud_prot_clash_idxs = self.check_for_clashes(prot_dists, self.pocket_radii)

        cloud_clash_idxs = cloud_lig_clash_idxs + cloud_prot_clash_idxs
        # get the idxs of conformers that are associated with clashes
        clash_conf_ids = self.retrieve_clashing_confs(cloud_clash_idxs)
        # get mol ids where every conf is clashing
        clash_mol_ids = self.retrieve_clashing_mols(clash_conf_ids)

        ################################# check for potential hbonds ################################
        interactions = None
        if self.check_for_ints:
            pdon_interactions, ldon_interactions, interacting_prot_ids, \
            interacting_conf_ids = self.check_for_potential_interactions(prot_dists, clash_conf_ids)
            interacting_mol_ids = set(self.retrieve_mol_from_conf(interacting_conf_ids))
            interactions = {'pdon_interactions': pdon_interactions,
                            'ldon_interactions': ldon_interactions,
                            'interacting_prot_ids': interacting_prot_ids,
                            'interacting_conf_ids': interacting_conf_ids,
                            'interacting_mol_ids': interacting_mol_ids}

        ####################################### write results ######################################
        self.results[vector.vector_id] = {'anchor_atom': vector.anchor_atom_idx,
                                          'replace_atoms': vector.replace_atom_idxs,
                                          'anchor_coord': vector.anchor_coord,
                                          'replace_coord': vector.replace_coord,
                                          'clashing_conf_ids': clash_conf_ids,
                                          'clashing_mol_ids': clash_mol_ids}
        if self.check_for_ints:
            self.results[vector.vector_id].update(interactions)

    def retrieve_clashing_confs(self, cloud_clash_idxs):
        """

        :param cloud_clash_idxs:
        :return:
        """
        # for each clashing cluster point, retrieve the associated conformers
        clashing_confs = set()
        for cloud_idx in cloud_clash_idxs:
            confs = self.cloud.cluster_points[cloud_idx].conf_ids
            clashing_confs.update(confs)
        return list(clashing_confs)

    def retrieve_clashing_mols(self, clashing_confs):
        """

        :param clashing_confs:
        :return:
        """
        # clashing_mol_ids = [self.cloud.conf_to_mol[conf] for conf in clashing_confs]
        clashing_mol_ids = self.retrieve_mol_from_conf(clashing_confs)
        clashing_mols = [mol_id for mol_id in set(clashing_mol_ids) if clashing_mol_ids.count(mol_id) == self.cloud.mol_conf_counts[mol_id]]
        return clashing_mols

    def retrieve_mol_from_conf(self, confs):
        """

        :param confs:
        :return:
        """
        mol_ids = [self.cloud.conf_to_mol[conf] for conf in confs]
        return mol_ids

    def evaluate_all_vectors(self):
        """

        :return:
        """
        if not self.vectors:
            self.get_vectors()
        for i, vector in enumerate(self.vectors):
            if self.verbose:
                print(f'Evaluating vector pair {i+1}/{len(self.vectors)}: {vector.anchor_atom_idx},{vector.replace_atom_idxs}')
            self.evaluate(vector)

    def binary_scorer(self):
        """

        :return:
        """
        if not self.results:
            self.evaluate_all_vectors()

        self.scored = {}

        for vector_id in self.results:
            # get basic info for easy access
            info = {'anchor_atom': self.results[vector_id]['anchor_atom'],
                    'replace_atoms': self.results[vector_id]['replace_atoms'],
                    'anchor_coord': self.results[vector_id]['anchor_coord'],
                    'replace_coord': self.results[vector_id]['replace_coord']}
            # check at least X elaborations can be added (at least one conf not clashing)
            if (self.total_mol_elabs - len(self.results[vector_id]['clashing_mol_ids'])) >= self.min_mols_added \
                and (self.total_mol_elabs - len(self.results[vector_id]['clashing_mol_ids'])) <= self.max_mols_added:
                clash_check = True
            else:
                clash_check = False
            all_checks = [clash_check]
            self.scored[vector_id] = {'clash_check': clash_check}

            # check at least X potential interactions can be reached
            if self.check_for_ints:
                if len(self.results[vector_id]['interacting_prot_ids']) >= self.min_ints_reached:
                    int_check = True
                else:
                    int_check = False
                all_checks = [clash_check, int_check]
                self.scored[vector_id]['int_check'] = int_check

            passing = len(all_checks) == sum(all_checks)
            self.scored[vector_id]['pass'] = passing

            # add basic info for easy access
            self.scored[vector_id].update(info)


