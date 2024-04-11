
from argparse import ArgumentParser

import numpy as np
from sklearn.cluster import DBSCAN


def save_npy(arr, fname):
    with open(fname, 'wb') as f:
        np.save(f, arr)


def main():
    parser = ArgumentParser()
    parser.add_argument('--coords_file')
    parser.add_argument('--output_file')
    parser.add_argument('--min_samples', default=1000, type=int)
    parser.add_argument('--eps', default=1.0, type=float)
    parser.add_argument('--n_jobs', default=1, type=int)
    args = parser.parse_args()

    coords = np.load(args.coords_file)

    clustering = DBSCAN(eps=args.eps,
                        min_samples=args.min_samples,
                        metric='euclidean',
                        n_jobs=args.n_jobs).fit(coords)
    print(len(clustering.components_), 'clustering components')
    save_npy(clustering.components_, args.output_file)


if __name__ == "__main__":
    main()
