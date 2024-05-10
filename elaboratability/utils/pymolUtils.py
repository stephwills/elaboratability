'''
http://pymolwiki.org/index.php/FindSurfaceResidues
'''

from __future__ import print_function
from pymol import cmd
import elaboratability.utils.processConfig as config

def findSurfaceAtoms(pdb_file, selection="all", cutoff=2.5, quiet=1):
    """
    DESCRIPTION
        Finds those atoms on the surface of a protein
        that have at least 'cutoff' exposed A**2 surface area.
    USAGE
        findSurfaceAtoms [ selection, [ cutoff ]]
    SEE ALSO
        findSurfaceResidues
    """
    cmd.reinitialize()
    cmd.load(pdb_file)
    cutoff, quiet = float(cutoff), int(quiet)
    tmpObj = cmd.get_unused_name("_tmp")
    cmd.create(tmpObj, "(" + selection + ") and polymer", zoom=0)
    cmd.set("dot_solvent", 1, tmpObj)
    cmd.get_area(selection=tmpObj, load_b=1)
    # threshold on what one considers an S"exposed" atom (in A**2):
    cmd.remove(tmpObj + " and b < " + str(cutoff))
    selName = cmd.get_unused_name("exposed_atm_")
    cmd.select(selName, "(" + selection + ") in " + tmpObj)
    cmd.delete(tmpObj)
    if not quiet:
        print("Exposed atoms are selected in: " + selName)
    surface_atoms = {'IDs': []}
    cmd.iterate_state(1, selName, "IDs.append(ID)", space=surface_atoms)
    return surface_atoms


def findHDonorsAcceptors(pdb_file):
    cmd.reinitialize()
    cmd.load(pdb_file)
    hbonders = {"acceptor_coords": [], "acceptor_IDs": [], "donor_coords": [], "donor_IDs": []}
    cmd.select("acc", "acceptors")
    cmd.iterate_state(1, "acc", "acceptor_coords.append([x,y,z])", space=hbonders)
    cmd.iterate_state(1, "acc", "acceptor_IDs.append(ID)", space=hbonders)
    cmd.select("don", "donors")
    cmd.iterate_state(1, "don", "donor_coords.append([x,y,z])", space=hbonders)
    cmd.iterate_state(1, "don", "donor_IDs.append(ID)", space=hbonders)
    return hbonders


def findAromaticAtoms(pdb_file):
    cmd.reinitialize()
    cmd.load(pdb_file)
    aromatics = {"aromatic_coords": [], "aromatic_IDs": [], "aromatic_resis": []}
    cmd.select("aromatics", "(resn phe+tyr+trp+his)")
    cmd.select("byring aromatics")
    cmd.set_name("sele", "aromatic_rings")
    cmd.iterate_state(1, "aromatic_rings", "aromatic_coords.append([x,y,z])", space=aromatics)
    cmd.iterate_state(1, "aromatic_rings", "aromatic_IDs.append(ID)", space=aromatics)
    cmd.iterate_state(1, "aromatic_rings", "aromatic_resis.append(resi)", space=aromatics)
    return aromatics


def save_close_protein_atoms(pdb_file, ligand_coords, dist_threshold=8, output_fname=None, save_pocket_file=None,
                             element_names=config.ELEMENT_NAMES):
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
    from elaboratability.utils.utils import get_distance
    import numpy as np
    coords = {"coords": [], "IDs": [], "elems": []}
    cmd.reinitialize()
    cmd.load(pdb_file, "prot")
    cmd.iterate_state(1, "all", "coords.append([x,y,z])", space=coords)
    cmd.iterate_state(1, "all", "IDs.append(ID)", space=coords)
    cmd.iterate_state(1, "all", "elems.append(elem)", space=coords)

    # select those protein atoms that are within a certain distance of molecule
    pocket_coords = {"coords": [], "IDs": [], "elems": []}
    for prot_coord, id, elem in zip(coords["coords"], coords["IDs"], coords["elems"]):
        if elem in element_names:
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

    return pocket_coords



def create_pymol_sesh(mol_file, pdb_file, eval_data, pse_fname, n_mols=302):
    """
    :param mol_file:
    :param pdb_file:
    :param clash_data: dictionary with H idxs as keys and number of clashes as values
    :param pse_fname:
    :return:
    """
    from elaboratability.utils.geometricUtils import get_coords_of_vector
    from elaboratability.utils.drawArrows import cgo_arrow
    from pymol import cmd
    from rdkit import Chem

    hs_mol = Chem.AddHs(Chem.MolFromMolFile(mol_file), addCoords=True)
    cmd.reinitialize()
    cmd.load(mol_file, 'LIG')
    cmd.load(pdb_file, 'PROT')

    for vector in eval_data:
        ### make arrow
        anchor_atom = eval_data[vector]['anchor_atom']
        replaced_atom = eval_data[vector]['replaced_atom']
        anchor_coords, replaced_coords = get_coords_of_vector(hs_mol, anchor_atom, replaced_atom)

        clash_mol_count = len(eval_data[vector]['clashing_mol_ids'])

        if clash_mol_count > (n_mols * 0.9):
            color = 'black'

        if (n_mols * 0.75) < clash_mol_count <= (n_mols * 0.9):
            color = 'tv_red'

        if (n_mols * 0.5) < clash_mol_count <= (n_mols * 0.75):
            color = 'orange'

        if (n_mols * 0.25) < clash_mol_count <= (n_mols * 0.5):
            color = 'brightorange'

        if (n_mols * 0.1) <= clash_mol_count <= (n_mols * 0.25):
            color = 'brightorange'

        if clash_mol_count < (n_mols * 0.1):
            color = 'green'

        cgo_arrow(list(anchor_coords), list(replaced_coords), color=color, name=f"{anchor_atom}_arrow", radius=0.1)
        # ### save clash
        # clash_atom_name = 'clash-' + str(h_idx)
        # clash_atoms = eval_data[h_idx][3]
        # if len(clash_atoms) > 0:
        #     cmd.select(clash_atom_name, "id " + '+'.join(map(str, clash_atoms)))

        int_atoms = eval_data[vector]['int_prot_don_ids'] + eval_data[vector]['int_prot_acc_ids']
        int_atom_name = 'int-' + str(anchor_atom)
        if len(int_atoms) > 0:
            cmd.select(int_atom_name, "id " + '+'.join(map(str, int_atoms)))

    cmd.show('surface', 'PROT')
    cmd.hide('cartoon', 'PROT')
    cmd.set('transparency', '0.5')
    cmd.save(pse_fname)
