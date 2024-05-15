
class Vector():

    def __init__(self, anchor_atom_idx, replace_atom_idxs, anchor_coord, replace_coord, rotated: bool, is_hyd: bool,
                 vector_id=None):
        """

        :param anchor_atom_idx:
        :param replace_atom_idxs:
        :param anchor_coord:
        :param replace_coord:
        :param rotated:
        :param is_hyd:
        :param vector_id:
        """
        self.vector_id = vector_id
        # a single idx
        self.anchor_atom_idx = anchor_atom_idx
        # may be multiple possible indices that we'll treat equally (H atoms can be rotated)
        self.replace_atom_idxs = replace_atom_idxs
        self.anchor_coord = anchor_coord
        self.replace_coord = replace_coord
        self.rotated = rotated
        self.is_hyd = is_hyd
