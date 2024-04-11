'''
http://pymolwiki.org/index.php/FindSurfaceResidues
'''

from __future__ import print_function
from pymol import cmd


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
    # threshold on what one considers an "exposed" atom (in A**2):
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
