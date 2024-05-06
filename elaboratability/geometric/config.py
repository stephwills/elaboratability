import numpy as np
from elaboratability.utils.notebookUtils import load_json

atom_numbers = {
    'C': 6,
    'N': 7,
    'O': 8,
    'S': 16,
    'F': 9,
    'P': 15,
    'Cl': 17,
    'Mg': 12,
    'Br': 35
}

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

conf_to_mol_dict = load_json('/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/coords/conf_to_mol_dict.json')
conf_to_mol_dict = {int(key): conf_to_mol_dict[key] for key in conf_to_mol_dict}
mol_conf_counts = load_json('/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/coords/mol_conf_counts.json')
mol_conf_counts = {int(key): mol_conf_counts[key] for key in mol_conf_counts}
cluster_coordinates = np.load('/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/coords/cluster_coordinates.npy')
cluster_elems = load_json('/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/coords/cluster_elems.json')
cluster_radii = load_json('/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/coords/cluster_radii.json')
cluster_to_conf_dict = load_json('/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/coords/cluster_to_conf_dict.json')
cluster_to_conf_dict = {int(key): cluster_to_conf_dict[key] for key in cluster_to_conf_dict}
atom_cluster_labels = load_json('/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/coords/atom_cluster_labels.json')
atom_confids = load_json('/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/coords/atom_confids.json')
atom_molids = load_json('/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/coords/atom_molids.json')
atom_is_acceptor = load_json('/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/coords/atom_is_acceptor.json')
atom_is_donor = load_json('/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/coords/atom_is_donor.json')

clust_is_donor = load_json('/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/coords/clust_is_donor.json')
clust_is_donor = {int(key): clust_is_donor[key] for key in clust_is_donor}
clust_is_acceptor = load_json('/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/coords/clust_is_acceptor.json')
clust_is_acceptor = {int(key): clust_is_acceptor[key] for key in clust_is_acceptor}
clust_donor_confs = load_json('/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/coords/clust_donor_confs.json')
clust_donor_confs = {int(key): clust_donor_confs[key] for key in clust_donor_confs}
clust_acceptor_confs = load_json('/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/coords/clust_acceptor_confs.json')
clust_acceptor_confs = {int(key): clust_acceptor_confs[key] for key in clust_acceptor_confs}

cloud_attach_coords=np.array([-0.75200022,0.,0.])
cloud_adj_coords=np.array([0.75200022,0.,0.])

clash_cutoff = 2.331
hbond_cutoff = 3