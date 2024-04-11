import os

import config
import numpy as np
from addElabsToMol import get_coords_of_vector, get_h_idxs
from alignConformersToRef import (calc_translation, get_rotated_coord,
                                  get_rotation_matrix_for_mol,
                                  translate_coords)
from chem_utils import *
from joblib import Parallel, delayed
from notebook_utils import *
from rdkit import Chem
from scipy.spatial.distance import cdist
from tqdm import tqdm


def get_coords_as_np_array(mol, idxs=None, ignoreDummy=False):
    conf = mol.GetConformer()
    if not ignoreDummy:
        all_coords = np.array(conf.GetAtomPosition(0))

        for atom in range(1, mol.GetNumAtoms()):
            #print(atom)
            all_coords = np.vstack((all_coords, np.array(conf.GetAtomPosition(atom))))
        # idxs = list(range(mol.GetNumAtoms()))
    else:
        # idxs = [atom.GetIdx() for atom in mol.GetAtoms() if mol.GetAtomicNum() != 0]
        all_coords = np.array(conf.GetAtomPosition(idxs[0]))
        if len(idxs) == 1:
            all_coords = np.array([conf.GetAtomPosition(idxs[0])])
        if len(idxs) > 1:
            all_coords = np.array(conf.GetAtomPosition(idxs[0]))
            for atom in idxs[1:]:
                all_coords = np.vstack((all_coords, np.array(conf.GetAtomPosition(atom))))
    return all_coords


def align_cloud_to_vector(h_coords, neigh_coords, cloud_coords, cloud_attach_coords, cloud_adj_coords):
    # attach coords and adj coords should just be the anchor mol coordinates used to generate the cloud
    translation = calc_translation(h_coords, cloud_adj_coords)  # ref coord, query coord
    cloud_adj_coords = h_coords

    cloud_attach_coords = translate_coords(translation, cloud_attach_coords)
    all_coords = [translate_coords(translation, coords) for coords in cloud_coords]
    rotat_mat = get_rotation_matrix_for_mol(h_coords, neigh_coords, cloud_attach_coords)
    all_coords = [get_rotated_coord(cloud_adj_coords, coords, rotat_mat) for coords in all_coords]
    return all_coords


def save_close_protein_atoms(pdb_file, ligand_coords, dist_threshold=8, output_fname=None, save_pocket_file=None):
    """
    Get the coordinates of 'pocket atoms' (within a certain distance) to limit the number of distances that have to
    be calculated later. Optionally save as a standalone pdb file.

    :param pdb_file:
    :param ligand_coords:
    :param dist_threshold:
    :param output_fname:
    :param savePocketFile:
    :return:
    """
    from pymol import cmd
    coords = {"coords": [], "IDs": [], "elems": []}
    cmd.reinitialize()
    cmd.load(pdb_file, "prot")
    cmd.iterate_state(1, "all", "coords.append([x,y,z])", space=coords)
    cmd.iterate_state(1, "all", "IDs.append(ID)", space=coords)
    cmd.iterate_state(1, "all", "elems.append(elem)", space=coords)

    # select those protein atoms that are within a certain distance and angle of the exit vector
    pocket_coords = {"coords": [], "IDs": [], "elems": []}
    for prot_coord, id, elem in zip(coords["coords"], coords["IDs"], coords["elems"]):
        if id not in pocket_coords["IDs"]:
            for ligand_coord in ligand_coords:
                if id not in pocket_coords["IDs"]:
                    dist = get_distance(ligand_coord, np.array(prot_coord))
                    # print(dist, dist_threshold)
                    if dist <= dist_threshold:
                        pocket_coords["IDs"].append(id)
                        pocket_coords["coords"].append(prot_coord)
                        pocket_coords["elems"].append(elem)

    # save a pdb file with just the selected protein atoms
    if save_pocket_file:
        if len(pocket_coords["IDs"]) > 0:
            select_IDs = pocket_coords["IDs"]
            ids_string = "id "
            for num in select_IDs[: len(select_IDs) - 1]:
                ids_string += str(num)
                ids_string += "+"
            ids_string += str(select_IDs[-1])
            cmd.select("pocket", ids_string)
            cmd.save(output_fname, "pocket")

    return coords, pocket_coords


def evaluate_with_cloud_coords(mol, h_idx, ligand_coords, ligand_radii,
                               cluster_coords, cluster_radii,
                               cluster_to_conf_dict, conf_to_mol_dict, mol_conf_counts,
                               prot_coords, prot_radii, prot_ids,
                               attach_coords, adj_coords,
                               # adj_coords, attach_coords,

                               prot_is_donor, prot_is_acceptor,
                               clust_is_donor, clust_is_acceptor,
                               clust_donor_confs, clust_acceptor_confs,
                               clash_cutoff=2.331,
                               hbond_cutoff=None,
                               check_ints=True):

    # align clustered cloud to the vector
    h_coords, neigh_coords = get_coords_of_vector(mol, h_idx)  # get coords of vector atoms
    aligned_coords = align_cloud_to_vector(h_coords, neigh_coords, cluster_coords, attach_coords, adj_coords)

    # # TODO: check
    # check if elabs clash with the ligand itself
    lig_dists = cdist(ligand_coords, aligned_coords)
    lig_clash_thresh = np.where(lig_dists < clash_cutoff)

    lig_clash_idxs = []
    cluster_lig_clash_idxs = []

    for lig_idx, dec_idx in zip(lig_clash_thresh[0], lig_clash_thresh[1]):
        cutoff = 0.63 * (ligand_radii[lig_idx] + cluster_radii[dec_idx])
        if lig_dists[lig_idx, dec_idx] < cutoff:
            lig_clash_idxs.append(lig_idx)
            cluster_lig_clash_idxs.append(dec_idx)

    # calculate the dists between the aligned clustered cloud and the protein pocket atoms
    dists = cdist(prot_coords, aligned_coords)

    # get the clusters that are clashing/not clashing
    clash_thresh = np.where(dists < clash_cutoff)

    n_clashes = 0
    cluster_clash_idxs = []  # the indices of clustered coordinates that result in a clash
    prot_clash_idxs = []  # the indices of prot coordinates that result in a clash

    for prot_idx, dec_idx in zip(clash_thresh[0], clash_thresh[1]):
        cutoff = 0.63 * (prot_radii[prot_idx] + cluster_radii[dec_idx])

        if dists[prot_idx, dec_idx] < cutoff:
            n_clashes += 1
            prot_clash_idxs.append(prot_idx)
            cluster_clash_idxs.append(dec_idx)

    # TODO: check
    # add the clashing idxs for the clusters that clash with the ligand
    # cluster_clash_idxs = cluster_clash_idxs + cluster_lig_clash_idxs


    # get the conformers associated with the clusters
    clash_conf_ids = set()
    for idx in cluster_clash_idxs:
        clash_conf_ids.update(cluster_to_conf_dict[idx])
    clash_conf_ids = list(clash_conf_ids)

    print(len(clash_conf_ids), 'clashing conformers')
    # get the mol ids that only result in clashing conformers
    # print(clash_conf_ids[:5])
    clash_mol_ids = [conf_to_mol_dict[confid] for confid in set(clash_conf_ids)]
    all_clashing_mol_ids = [mol_id for mol_id in set(clash_mol_ids) if clash_mol_ids.count(mol_id) == mol_conf_counts[(mol_id)]]

    if not check_ints:

        return all_clashing_mol_ids, clash_conf_ids

    else:

        # get all the dists that are not clashing but are below hbond threshold
        hbond_thresh = np.where(dists < hbond_cutoff)
        prot_don_idxs = set()
        cluster_acc_idxs = []

        prot_acc_idxs = set()
        cluster_don_idxs = []

        for prot_idx, dec_idx in zip(hbond_thresh[0], hbond_thresh[1]):
            # record POTENTIAL h bonds where protein is donor
            if prot_is_donor[prot_idx]:  # if the prot atom is donor
                if clust_is_acceptor[dec_idx]:

                    for conf in clust_acceptor_confs[dec_idx]:
                        if conf not in clash_conf_ids:
                            cluster_acc_idxs.append(dec_idx)
                            prot_don_idxs.add(prot_idx)
                            break

            if prot_is_acceptor[prot_idx]:
                if clust_is_donor[dec_idx]:

                    for conf in clust_donor_confs[dec_idx]:
                        if conf not in clash_conf_ids:
                            cluster_don_idxs.append(dec_idx)
                            prot_acc_idxs.add(prot_idx)
                            break

        int_prot_don_ids = [prot_ids[idx] for idx in prot_don_idxs]
        int_prot_acc_idxs = [prot_ids[idx] for idx in prot_acc_idxs]

        return all_clashing_mol_ids, clash_conf_ids, int_prot_don_ids, int_prot_acc_idxs, aligned_coords, h_coords, neigh_coords, ligand_coords

def get_all_coords(mol):
    conf = mol.GetConformer()
    coords = [np.array(conf.GetAtomPosition(idx)) for idx in range(mol.GetNumAtoms())]
    return coords


def get_all_coords_and_radii(mol):
    # from config import atom_radii
    atom_radii = {
        6: 1.70,
        7: 1.55,
        8: 1.52,
        16: 1.80,
        9: 1.47,
        15: 1.80,
        17: 1.75,
        12: 1.73,
        35: 1.85
    }
    conf = mol.GetConformer()
    coords = [np.array(conf.GetAtomPosition(idx)) for idx in range(mol.GetNumAtoms())]
    molnums = [mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in range(mol.GetNumAtoms())]
    radii = [atom_radii[num] for num in molnums]
    return coords, radii


def evaluate_mol_against_cloud(mol,
                               pdb_file,
                               save_pocket_file=False,
                               pocket_file_name=None,
                               pocket_dist=8,
                               conf_to_mol_dict=config.conf_to_mol_dict,
                               mol_conf_counts=config.mol_conf_counts,
                               cluster_coords=config.cluster_coordinates,
                               cluster_radii=config.cluster_radii,
                               cluster_to_conf_dict=config.cluster_to_conf_dict,
                               cloud_attach_coords=np.array([0.75200022,0.,0.]),
                               cloud_adj_coords=np.array([-0.75200022,0.,0.]),
                               clash_cutoff=config.clash_cutoff,
                               hbond_cutoff=config.hbond_cutoff,
                               clust_is_donor=config.clust_is_donor,
                               clust_is_acceptor=config.clust_is_acceptor,
                               clust_donor_confs=config.clust_donor_confs,
                               clust_acceptor_confs=config.clust_acceptor_confs,
                               cpus=4,
                               check_ints=True
                               ):
    ligand_coords, ligand_radii = get_all_coords_and_radii(mol)

    # get the pocket coords and atom ids
    prot_data, pocket_data = save_close_protein_atoms(pdb_file,
                                                      ligand_coords,
                                                      dist_threshold=pocket_dist,
                                                      output_fname=pocket_file_name,
                                                      save_pocket_file=save_pocket_file)
    pocket_coords, pocket_ids, pocket_elems = np.array(pocket_data['coords']), pocket_data['IDs'], pocket_data['elems']
    pocket_radii = [config.atom_radii[config.atom_numbers[elem]] for elem in pocket_elems]

    # we will treat each H bond as a potential expansion vector
    mol, h_idxs = get_h_idxs(mol, addHs=True)

    prot_is_donor=None
    prot_is_acceptor=None
    if check_ints:
        # process the protein pocket to find H donors and acceptors
        from pymolFunctions import findSurfaceAtoms, findHDonorsAcceptors
        # get the IDs of surface atoms from the full pdb file
        surface_ids = findSurfaceAtoms(pdb_file)['IDs']

        # get the IDs of hbond donors and acceptors
        hbonder_data = findHDonorsAcceptors(pdb_file)
        hbond_donor_ids = hbonder_data["donor_IDs"]
        hbond_donor_ids = get_intersect(hbond_donor_ids, pocket_ids)  # select only those in pocket
        hbond_donor_ids = get_intersect(hbond_donor_ids, surface_ids)  # select only those on surface

        hbond_acceptor_ids = hbonder_data["acceptor_IDs"]
        hbond_acceptor_ids = get_intersect(hbond_acceptor_ids, pocket_ids)  # select only those in pocket
        hbond_acceptor_ids = get_intersect(hbond_acceptor_ids, surface_ids)  # select only those on surface

        # convert into a bool list
        prot_is_donor = [id in hbond_donor_ids for id in pocket_ids]
        prot_is_acceptor = [id in hbond_acceptor_ids for id in pocket_ids]

    # get data for each H index
    eval_data = Parallel(n_jobs=cpus, backend="multiprocessing")(
        delayed(evaluate_with_cloud_coords)(mol, h_idx, np.array(ligand_coords), ligand_radii,
                                            cluster_coords, cluster_radii,
                                            cluster_to_conf_dict, conf_to_mol_dict, mol_conf_counts,
                                            pocket_coords, pocket_radii, pocket_ids,
                                            cloud_attach_coords, cloud_adj_coords,
                                            prot_is_donor, prot_is_acceptor,
                                            clust_is_donor, clust_is_acceptor,
                                            clust_donor_confs, clust_acceptor_confs,
                                            clash_cutoff, hbond_cutoff, check_ints=True)
        for h_idx in tqdm(h_idxs, total=len(h_idxs), position=0, leave=True)
    )
    eval = {h_idx: data for h_idx, data in zip(h_idxs, eval_data)}
    return eval


def create_pymol_sesh(mol_file, pdb_file, eval_data, pse_fname):
    """
    :param mol_file:
    :param pdb_file:
    :param clash_data: dictionary with H idxs as keys and number of clashes as values
    :param pse_fname:
    :return:
    """
    from pymol import cmd
    from drawArrows import cgo_arrow
    from addElabsToMol import get_coords_of_vector
    from rdkit import Chem
    hs_mol = Chem.AddHs(Chem.MolFromMolFile(mol_file), addCoords=True)
    cmd.reinitialize()
    cmd.load(mol_file, 'LIG')
    cmd.load(pdb_file, 'PROT')
    for h_idx in eval_data:
        ### make arrow
        count = len(eval_data[h_idx][0])
        h_coords, neigh_coords = get_coords_of_vector(hs_mol, h_idx)
        if count >= 250:
            color = 'black'
        if 250 > count >= 200:
            color = 'tv_red'
        if 200 > count >= 150:
            color = 'orange'
        if 150 > count >= 100:
            color = 'brightorange'
        if 100 > count >= 50:
            color = 'yelloworange'
        if count < 50:
            color = 'green'

        cgo_arrow(list(neigh_coords), list(h_coords), color=color, name=f"{h_idx}_arrow", radius=0.1)


        # ### save clash
        # clash_atom_name = 'clash-' + str(h_idx)
        # clash_atoms = eval_data[h_idx][3]
        # if len(clash_atoms) > 0:
        #     cmd.select(clash_atom_name, "id " + '+'.join(map(str, clash_atoms)))

        int_atoms = eval_data[h_idx][2] + eval_data[h_idx][3]
        int_atom_name = 'int-' + str(h_idx)
        if len(int_atoms) > 0:
            cmd.select(int_atom_name, "id " + '+'.join(map(str, int_atoms)))

    cmd.show('surface', 'PROT')
    cmd.hide('cartoon', 'PROT')
    cmd.set('transparency', '0.5')
    cmd.save(pse_fname)
