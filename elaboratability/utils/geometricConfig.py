import numpy as np

ATOM_RADII = {
    6: 1.70,
    7: 1.55,
    8: 1.52,
    16: 1.80,
    9: 1.47,
    15: 1.80,
    17: 1.75,
    12: 1.73,
    35: 1.85,
    14: 2.1,
    11: 2.27,
    20: 2.31,
    26: 1.94,
    30: 1.39,
    19: 275
}
CLOUD_ATTACH_COORDS = np.array([0.75200022,0.,0.])
CLOUD_ADJ_COORDS = np.array([-0.75200022,0.,0.])
CLASH_CUTOFF = 2.331
HBOND_CUTOFF = 3
POCKET_DIST = 8.0
CHECK_LIG_CLASH = True
CHECK_FOR_INTS = True