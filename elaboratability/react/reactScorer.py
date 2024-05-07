
import elaboratability.utils.geometricConfig as config
import elaboratability.utils.processConfig as proConfig
import elaboratability.utils.reactConfig as reactConfig
import numpy as np
from elaboratability.geometric.scorer import Scorer
from elaboratability.react.createReaction import \
    filter_multiple_rxns_for_vector_with_aizynthfinder
from scipy.spatial.distance import cdist


class ReactionScorer(Scorer):

    def __init__(self, ligand, pdb_file, cloud, pocket_dist=config.POCKET_DIST, clash_cutoff=config.CLASH_CUTOFF,
                 hbond_cutoff=config.HBOND_CUTOFF, cloud_attach_coords=config.CLOUD_ATTACH_COORDS,
                 cloud_adj_coords=config.CLOUD_ADJ_COORDS, atom_dict=proConfig.ATOM_DICT, check_lig_clash=config.CHECK_LIG_CLASH,
                 check_for_ints=config.CHECK_FOR_INTS, total_mol_elabs=config.TOTAL_MOLS, total_confs=config.TOTAL_CONFS, min_prop_mols_added=config.MIN_PROP_MOLS_ADDED,
                 max_prop_mols_added=config.MAX_PROP_MOLS_ADDED, min_ints_reached=config.MIN_INTS_REACHED, aizynth_config=reactConfig.AIZYNTH_CONFIG, filter_model=reactConfig.FILTER_MODEL,
                 filter_cutoff=reactConfig.FILTER_CUTOFF, check_all_non_clashing=reactConfig.CHECK_ALL_NON_CLASHING):
        """

        :param aizynth_config:
        :param filter_model:
        :param filter_cutoff:
        """
        Scorer.__init__(self, ligand, pdb_file, cloud, pocket_dist, clash_cutoff, hbond_cutoff, cloud_attach_coords,
                        cloud_adj_coords, atom_dict, check_lig_clash, check_for_ints, total_mol_elabs, total_confs, min_prop_mols_added,
                        max_prop_mols_added, min_ints_reached)
        self.aizynth_config = aizynth_config
        self.filter_model = filter_model
        self.filter_cutoff = filter_cutoff
        self.check_all_non_clashing = check_all_non_clashing

    def check_for_potential_interactions(self, dists, clash_conf_ids):
        """

        :param dists:
        :param clash_conf_ids:
        :return:
        """
        pdon_interactions = []
        ldon_interactions = []

        interacting_prot_ids = []  # change to list from set so we can filter out later based on reactions
        interacting_conf_ids = []

        # self.prot_is_don, self.prot_is_acc, whether cluster point associated with don or acc
        thresh = np.where(dists < self.hbond_cutoff)
        for prot_idx, cloud_idx in zip(thresh[0], thresh[1]):
            # check for potential interactions where protein is donor
            if self.prot_is_don[prot_idx] and self.cloud.cluster_points[cloud_idx].acc_flag:
                # check the conformers are not clashing
                for conf in self.cloud.cluster_points[cloud_idx].conf_ids:
                    if conf not in clash_conf_ids:
                        pdon_interactions.append([prot_idx, cloud_idx, conf])
                        interacting_prot_ids.append(prot_idx)
                        interacting_conf_ids.append(conf)

            # check for potential interactions where protein is acceptor
            if self.prot_is_acc[prot_idx] and self.cloud.cluster_points[cloud_idx].don_flag:
                # check the conformers are not clashing
                for conf in self.cloud.cluster_points[cloud_idx].conf_ids:
                    if conf not in clash_conf_ids:
                        ldon_interactions.append([prot_idx, cloud_idx, conf])
                        interacting_prot_ids.append(prot_idx)
                        interacting_conf_ids.append(conf)

        return pdon_interactions, ldon_interactions, interacting_prot_ids, interacting_conf_ids


    def evaluate(self, anchor_atom, replaced_atom):
        """

        :param anchor_atom:
        :param replaced_atom:
        :return:
        """
        # get the coordinates of the other atoms in the ligand (not including the vector)
        ligand_atoms, ligand_coords, ligand_radii = self.get_ligand_data_for_eval(anchor_atom, replaced_atom)

        # align the coordinates of the cloud to the vector
        aligned_coords = self.align_coords(anchor_atom, replaced_atom)

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
            interacting_mol_ids = self.retrieve_mol_from_conf(interacting_conf_ids)

            ################################# check for reacting mols ################################
            interacting_mol_smis = [self.cloud.mol_smiles[idx] for idx in interacting_mol_ids]

            uniq_smis = list(set(interacting_mol_smis))
            interacting_mol_smi_idxs = [uniq_smis.index(smi) for smi in interacting_mol_smis]

            vector_is_hyd = self.vector_is_hyd[(anchor_atom, replaced_atom)]
            filter_res, filter_feas = filter_multiple_rxns_for_vector_with_aizynthfinder(
                self.ligand, anchor_atom, replaced_atom, vector_is_hyd, uniq_smis,
                self.aizynth_config, self.filter_model, self.filter_cutoff)

            res_list = [filter_res[idx] for idx in interacting_mol_smi_idxs]
            reacting_conf_ids = [conf_id for conf_id, res in zip(interacting_conf_ids, res_list) if res]
            reacting_mol_ids = [mol_id for mol_id, res in zip(interacting_mol_ids, res_list) if res]
            reacting_prot_ids = [prot_id for prot_id, res in zip(interacting_prot_ids, res_list) if res]

            interactions = {'pdon_interactions': pdon_interactions,
                            'ldon_interactions': ldon_interactions,
                            'interacting_prot_ids': set(interacting_prot_ids),
                            'interacting_conf_ids': set(interacting_conf_ids),
                            'interacting_mol_ids': set(interacting_mol_ids),
                            'reacting_conf_ids': set(reacting_conf_ids),
                            'reacting_mol_ids': set(reacting_mol_ids),
                            'reacting_prot_ids': set(reacting_prot_ids)}

        ####################################### write results ######################################
        self.results[(anchor_atom, replaced_atom)] = {'clashing_conf_ids': clash_conf_ids,
                                                      'clashing_mol_ids': clash_mol_ids}
        if self.check_for_ints:
            self.results[(anchor_atom, replaced_atom)].update(interactions)


    def binary_scorer(self):
        """

        :return:
        """
        if not self.results:
            self.evaluate_all_vectors()

        self.scored = {}

        for vector_pair in self.results:
            # check at least X elaborations can be added (at least one conf not clashing) AND can react
            if len(self.results[vector_pair]['clashing_mol_ids']) >= self.min_mols_added \
                    and len(self.results[vector_pair]['clashing_mol_ids']) <= self.max_mols_added:
                clash_check = True
            else:
                clash_check = False
            all_checks = [clash_check]
            self.scored[vector_pair] = {'clash_check': clash_check}

            if self.check_for_ints:
                if len(self.results[vector_pair]['reacting_prot_ids']) >= self.min_ints_reached:
                    int_check = True
                else:
                    int_check = False
                all_checks = [clash_check, int_check]
                self.scored[vector_pair]['int_check'] = int_check

            passing = len(all_checks) == sum(all_checks)
            self.scored[vector_pair]['pass'] = passing