AIZYNTH_CONFIG_FILE='/home/swills/Oxford/elaboratability/data/aizynthfinder/config.yml'
DECORATOR_FILE='/home/swills/Oxford/elaboratability/data/decs_dict.json'
MIN_DECORATOR_COUNTS=10000
STOCK='zinc'
EXPANSION_POLICY='uspto'
FILTER_POLICY='uspto'
VERBOSE=True
ELEMENT_NAMES=['N', 'O', 'C', 'S', 'F', 'P', 'Cl', 'Mg', 'Br']
ATOM_DICT = {
    'C': {'atomic_number': 6,
          'atomic_radii': 1.70},
    'N': {'atomic_number': 7,
          'atomic_radii': 1.55},
    'O': {'atomic_number': 8,
          'atomic_radii': 1.52},
    'S': {'atomic_number': 16,
          'atomic_radii': 1.80},
    'F': {'atomic_number': 9,
          'atomic_radii': 1.47},
    'P': {'atomic_number': 15,
          'atomic_radii': 1.80},
    'Cl': {'atomic_number': 17,
          'atomic_radii': 1.75},
    'Mg': {'atomic_number': 12,
          'atomic_radii': 1.73},
    'Br': {'atomic_number': 35,
          'atomic_radii': 1.85}
}
CLUSTERING_DIST_THRESHOLD = 0.5
CLUSTERED_CONFORMER_FILE = '/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/12out.sdf'
CLUSTERED_CONFORMER_JSON = '/home/swills/Oxford/elaboratability/notebooks/12_evalWithClusteredPoints/12out.json'
PROCESSED_DATA_FILE = '/home/swills/Oxford/elaboratability/data/processed_cloud_data.pkl'
USE_PYMOL = False