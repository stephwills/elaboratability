import pickle

import elaboratability.utils.processConfig as config
import numpy as np
from elaboratability.preprocessing.prepareCloud import get_all_coordinates
from elaboratability.utils.utils import load_json
from rdkit import Chem


class ClusterPoint():

    def __init__(self, coordinates, element, mol_ids, conf_ids, is_don, is_acc):

        self.coords = coordinates
        self.elem = element
        self.mol_ids = mol_ids
        self.conf_ids = conf_ids
        self.is_don = is_don
        self.is_acc = is_acc
        self.don_flag = False
        self.acc_flag = False
        if sum(self.is_don) > 0:
            self.don_flag = True
        if sum(self.is_acc) > 0:
            self.acc_flag = True


class ClusterCloud():

    def __init__(self, conf_file=config.CLUSTERED_CONFORMER_FILE, info_file=config.CLUSTERED_CONFORMER_JSON,
                 data_file=config.PROCESSED_DATA_FILE, reprocess_data=False):
        """

        :param conf_file:
        :param info_file:
        :param data_file:
        :param reprocess_data:
        """
        self.conf_file = conf_file
        self.info_file = info_file
        if reprocess_data:
            self.data_file = None
        else:
            self.data_file = data_file

        info_data = load_json(self.info_file)
        self.conformers = [mol for mol in Chem.SDMolSupplier(self.conf_file)]
        self.mol_ids = info_data['molIds']
        self.mol_smiles = info_data['smis']
        print(len(self.conformers), 'conformers loaded for', len(set(self.mol_ids)), 'molecules')
        self.conf_to_mol = {idx: mol_id for idx, mol_id in zip(range(len(self.conformers)), self.mol_ids)}
        self.mol_conf_counts = {mol_id: len(confs) for mol_id, confs in enumerate(info_data['confIds'])}

    def generate_cloud_data(self, new_data_file=None):
        """

        :param new_data_file:
        :return:
        """
        print('Processing conformers to create cloud')
        self.data = get_all_coordinates(self.conformers, self.mol_ids)

        if new_data_file:
            print('Writing to new data file', new_data_file)
            with open(new_data_file, 'wb') as handle:
                pickle.dump(self.data, handle)

    @staticmethod
    def _read_data_from_pickle(data_file):
        """

        :param data_file:
        :return:
        """
        with open(data_file, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def process_cloud(self):
        if not self.data_file:
            self.generate_cloud_data()
        else:
            print('Reading data from', self.data_file)
            self.data = self._read_data_from_pickle(self.data_file)

        self.cluster_points = []

        for element in self.data:
            centroids = self.data[element]['centroids']
            labels = self.data[element]['labels']
            coord_ids = list(range(len(self.data[element]['coords'])))

            mol_ids, conf_ids, atom_ids = self.data[element]['mol_ids'], self.data[element]['conf_ids'], self.data[element]['atom_ids']
            is_don, is_acc = self.data[element]['is_don'], self.data[element]['is_acc']

            for centroid_idx, centroid in enumerate(centroids):
                centroid_mol_ids = self._get_points_from_labels(mol_ids, labels, centroid_idx)
                centroid_conf_ids = self._get_points_from_labels(conf_ids, labels, centroid_idx)
                centroid_atom_ids = self._get_points_from_labels(atom_ids, labels, centroid_idx)
                centroid_coord_ids = self._get_points_from_labels(coord_ids, labels, centroid_idx)
                centroid_is_don = self._get_points_from_labels(is_don, labels, centroid_idx)
                centroid_is_acc = self._get_points_from_labels(is_acc, labels, centroid_idx)

                cluster = ClusterPoint(centroid,
                                       element,
                                       centroid_mol_ids,
                                       centroid_conf_ids,
                                       centroid_is_don,
                                       centroid_is_acc)
                self.cluster_points.append(cluster)

        self.coords = np.array([cluster_point.coords for cluster_point in self.cluster_points])

    @staticmethod
    def _get_points_from_labels(data, labels, idx):
        return [point for point, lab in zip(data, labels) if lab == idx]
