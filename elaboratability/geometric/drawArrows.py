
'''
http://pymolwiki.org/index.php/cgo_arrow

(c) 2013 Thomas Holder, Schrodinger Inc.

License: BSD-2-Clause
'''

from pymol import cmd, cgo, CmdException


def cgo_arrow(atom1='pk1', atom2='pk2', radius=0.5, gap=0.0, hlength=-1, hradius=-1,
              color='blue red', name=''):
    '''
DESCRIPTION
    Create a CGO arrow between two picked atoms.
ARGUMENTS
    atom1 = string: single atom selection or list of 3 floats {default: pk1}
    atom2 = string: single atom selection or list of 3 floats {default: pk2}
    radius = float: arrow radius {default: 0.5}
    gap = float: gap between arrow tips and the two atoms {default: 0.0}
    hlength = float: length of head
    hradius = float: radius of head
    color = string: one or two color names {default: blue red}
    name = string: name of CGO object
    '''
    from chempy import cpv

    radius, gap = float(radius), float(gap)
    hlength, hradius = float(hlength), float(hradius)

    try:
        color1, color2 = color.split()
    except:
        color1 = color2 = color
    color1 = list(cmd.get_color_tuple(color1))
    color2 = list(cmd.get_color_tuple(color2))

    def get_coord(v):
        if not isinstance(v, str):
            return v
        if v.startswith('['):
            return cmd.safe_list_eval(v)
        return cmd.get_atom_coords(v)

    xyz1 = get_coord(atom1)
    xyz2 = get_coord(atom2)
    normal = cpv.normalize(cpv.sub(xyz1, xyz2))

    if hlength < 0:
        hlength = radius * 3.0
    if hradius < 0:
        hradius = hlength * 0.6

    if gap:
        diff = cpv.scale(normal, gap)
        xyz1 = cpv.sub(xyz1, diff)
        xyz2 = cpv.add(xyz2, diff)

    xyz3 = cpv.add(cpv.scale(normal, hlength), xyz2)

    obj = [cgo.CYLINDER] + xyz1 + xyz3 + [radius] + color1 + color2 + \
          [cgo.CONE] + xyz3 + xyz2 + [hradius, 0.0] + color2 + color2 + \
          [1.0, 0.0]

    if not name:
        name = cmd.get_unused_name('arrow')

    return cmd.load_cgo(obj, name)


def create_pymol_sesh(mol_file, pdb_file, eval_data, pse_fname):
    """

    :param mol_file:
    :param pdb_file:
    :param clash_data: dictionary with H idxs as keys and number of clashes as values
    :param pse_fname:
    :return:
    """
    from addElabsToMol import get_coords_of_vector
    from rdkit import Chem
    hs_mol = Chem.AddHs(Chem.MolFromMolFile(mol_file), addCoords=True)
    cmd.reinitialize()
    cmd.load(mol_file, 'LIG')
    cmd.load(pdb_file, 'PROT')
    for h_idx in eval_data:
        ### make arrow
        count = eval_data[h_idx][0]
        h_coords, neigh_coords = get_coords_of_vector(hs_mol, h_idx)
        if count >= 50000:
            color = 'red'
        if 10000 < count < 50000:
            color = 'orange'
        if 1000 < count <= 10000:
            color = 'yellow'
        if count <= 1000:
            color = 'green'
        cgo_arrow(list(neigh_coords), list(h_coords), color=color, name=f"{h_idx}_arrow", radius=0.1)

        ### save clash
        clash_atom_name = 'clash-' + str(h_idx)
        clash_atoms = eval_data[h_idx][1]
        if len(clash_atoms) > 0:
            cmd.select(clash_atom_name, "id " + '+'.join(map(str, clash_atoms)))

        don_atoms = eval_data[h_idx][2]
        don_atom_name = 'don-' + str(h_idx)
        if len(don_atoms) > 0:
            cmd.select(don_atom_name, "id " + '+'.join(map(str, don_atoms)))

        acc_atoms = eval_data[h_idx][3]
        acc_atom_name = 'acc-' + str(h_idx)
        if len(acc_atoms) > 0:
            cmd.select(acc_atom_name, "id " + '+'.join(map(str, acc_atoms)))

        pi_atoms = eval_data[h_idx][4]
        pi_atom_name = 'arom-' + str(h_idx)
        if len(pi_atoms) > 0:
            cmd.select(pi_atom_name, "id " + '+'.join(map(str, pi_atoms)))

    cmd.show('surface', 'PROT')
    cmd.hide('cartoon', 'PROT')
    cmd.set('transparency', '0.5')
    cmd.save(pse_fname)


