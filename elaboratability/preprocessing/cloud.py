import pickle

import elaboratability.utils.processConfig as config
import numpy as np
from elaboratability.preprocessing.prepareCloud import get_all_coordinates
from elaboratability.utils.utils import load_json
from rdkit import Chem
from tqdm import tqdm


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
                 data_file=config.PROCESSED_DATA_FILE, reprocess_data=config.REPROCESS_DATA, cloud_file=config.CLOUD_FILE,
                 reprocess_cloud=config.REPROCESS_CLOUD):
        """

        :param conf_file:
        :param info_file:
        :param data_file:
        :param reprocess_data:
        """
        self.conf_file = conf_file
        self.info_file = info_file
        # whether to extract coordinates from conformers and perform clustering
        # saves as data_file
        self.data_file = data_file
        self.reprocess_data = reprocess_data
        # whether to reprocess data_file into structure that can be loaded directly to cloud
        # loading from data_file directly can be slow for many cluster points
        self.cloud_file = cloud_file
        self.reprocess_cloud = reprocess_cloud

        info_data = load_json(self.info_file)
        self.conformers = [mol for mol in Chem.SDMolSupplier(self.conf_file)]
        self.mol_ids = info_data['molIds']
        self.mol_smiles = info_data['smis']
        print(len(self.conformers), 'conformers loaded for', len(set(self.mol_ids)), 'molecules')
        self.conf_to_mol = {idx: mol_id for idx, mol_id in zip(range(len(self.conformers)), self.mol_ids)}
        self.mol_conf_counts = {mol_id: len(confs) for mol_id, confs in enumerate(info_data['confIds'])}

    def process_cloud(self):
        """

        :return:
        """
        if self.reprocess_data:
            self.generate_cloud_data()
        else:
            print('Reading data from', self.data_file)
            self.data = self._read_data_from_pickle(self.data_file)

        if self.reprocess_cloud:
            print('Reprocessing cloud to', self.cloud_file)
            self._process_to_cloud()
            self.load_processed_cloud()

        else:
            print('Reading processed cloud from', self.cloud_file)
            self.load_processed_cloud()

    def generate_cloud_data(self):
        """

        :param new_data_file:
        :return:
        """
        print('Processing conformers to create cloud')
        self.data = get_all_coordinates(self.conformers, self.mol_ids)

        print('Writing to new data file', self.data_file)
        with open(self.data_file, 'wb') as handle:
            pickle.dump(self.data, handle)

    def load_processed_cloud(self):
        """

        :return:
        """
        cloud_data = self._read_data_from_pickle(self.cloud_file)
        self.cluster_points = []

        for data_point in cloud_data:
            cluster = ClusterPoint(data_point['centroid'],
                                   data_point['element'],
                                   data_point['centroid_mol_ids'],
                                   data_point['centroid_conf_ids'],
                                   data_point['centroid_is_don'],
                                   data_point['centroid_is_acc'])
            self.cluster_points.append(cluster)

        self.coords = np.array([cluster_point.coords for cluster_point in self.cluster_points])

    def _process_to_cloud(self):
        """

        :return:
        """
        cloud_data = []

        for element in self.data:
            print('Reading element', element)
            centroids = self.data[element]['centroids']
            labels = self.data[element]['labels']
            coord_ids = list(range(len(self.data[element]['coords'])))

            mol_ids, conf_ids, atom_ids = self.data[element]['mol_ids'], self.data[element]['conf_ids'], \
                                          self.data[element]['atom_ids']
            is_don, is_acc = self.data[element]['is_don'], self.data[element]['is_acc']

            for centroid_idx, centroid in tqdm(enumerate(centroids), total=len(centroids), position=0, leave=True):
                centroid_mol_ids = self._get_points_from_labels(mol_ids, labels, centroid_idx)
                centroid_conf_ids = self._get_points_from_labels(conf_ids, labels, centroid_idx)
                centroid_atom_ids = self._get_points_from_labels(atom_ids, labels, centroid_idx)
                centroid_coord_ids = self._get_points_from_labels(coord_ids, labels, centroid_idx)
                centroid_is_don = self._get_points_from_labels(is_don, labels, centroid_idx)
                centroid_is_acc = self._get_points_from_labels(is_acc, labels, centroid_idx)

                cluster_dict = {'centroid': centroid,
                                'element': element,
                                'centroid_mol_ids': centroid_mol_ids,
                                'centroid_conf_ids': centroid_conf_ids,
                                'centroid_is_don': centroid_is_don,
                                'centroid_is_acc': centroid_is_acc}
                cloud_data.append(cluster_dict)

        print('Writing to new cloud file', self.cloud_file)
        with open(self.cloud_file, 'wb') as handle:
            pickle.dump(cloud_data, handle)

    @staticmethod
    def _read_data_from_pickle(data_file):
        """

        :param data_file:
        :return:
        """
        with open(data_file, 'rb') as handle:
            data = pickle.load(handle)
        return data

    @staticmethod
    def _get_points_from_labels(data, labels, idx):
        return [point for point, lab in zip(data, labels) if lab == idx]
