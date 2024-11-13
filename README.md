# Elaboratability evaluator
 
Contains code for initial method to evaluate the 'elaboratability' of a compound.

## Background

Offers a way of evaluating the elaboratability of a compound considering both the chemical and geometric aspects. Possible growth vectors are identified (including any hydrogen atom or any terminal atom, where either the hydrogen or terminal atom is replaced by the decorator during elaboration); each vector undergoes evaluation to determine whether it is an elaboratable vector or not.

### Geometric evaluation
To evaluate the vectors, a 'cloud' representing the possible elaboration space is aligned with the vectors and the possible clashes between the cloud and the protein/ligand atoms are identified. You can also calculate distances between possible HBA/HBD atoms in the cloud and the protein to identify potential interaction opportunities. 

### Chemical evaluation
To evaluate whether addition of a decorator is chemically feasible, the AiZynthFinder 'reaction filter network' (QuickKerasFilter) is used. A reaction SMILES is created, solely from the molecule being evaluated, the decorator and the product molecule after elaboration and QuickKerasFilter is used to judge whether it is feasible or not. This provides a proxy of elaboratability and could be improved later!

### Elaboration 'cloud'

The elaboration 'cloud' consists of a bunch of decorators taken from the LibINVENT dataset (of scaffolds and decorators generated from ChEMBL), pre-generating multiple conformers for each of them, aligning them all according to the attachment vector, rotating them, and clustering the resulting multiple rotated conformers (this is in `preprocessing/rotations.py`).

To generate the cloud, the atoms are extracted from the clustered conformers for each element type, clustered using agglomerative clustering to create a ClusterPoint object, and assembled to a ClusterCloud object. Each cluster point is associated with a coordinate, element, list of mol IDs (representing which decorators it is associated with), list of conformer IDs (every conformer across all decorators is indexed), list of whether the associated atoms that contributed to that clustered point are donors or acceptors and an acceptor or donor bool flag according to whether there is any donor or acceptor associated with it.

## Install

To install the required packages:
```
conda create -n elab python=3.9 -y
conda activate elab
conda install -c conda-forge rdkit -y
conda install -c conda-forge numpy -y
conda install -c conda-forge sklearn -y
conda install -c conda-forge scipy -y
pip install joblib tqdm aizynthfinder
mkdir ${LOCAL_DIR}/aizynthfinder
download_public_data ${LOCAL_DIR}/aizynthfinder
conda install -c conda-forge pymol-open-source -y
```

## Usage

### Generate cloud

Use `preprocessing/rotations.py` to take in a json file containing a dictionary of decorators and their frequency of occurrence and generate a bunch of conformers that have been aligned, rotated and clustered. The resulting decorators can be used to generate the 'cloud'.

The ClusterCloud object in `preprocessing/cloud.py` is then used to take in these files, process them to get the cloud, and save the processed data to avoid re-running this again.

E.g. if already generated, this will load the cloud (decorators have been filtered for those that occur at least 10,000 times):

```
import os
from elaboratability.preprocessing.cloud import ClusterCloud

conformers_dir = 'data/conformers/conformers'
processed_cloud_dir = 'data/conformers/processed'
cloud = ClusterCloud(conf_file=os.path.join(conformers_dir, 'confs_10000.sdf'),
                     info_file=os.path.join(conformers_dir, 'confs_10000.json'),
                     data_file=os.path.join(processed_cloud_dir, 'confs_10000.json'),
                     cloud_file=os.path.join(processed_cloud_dir, 'cloud_10000.pkl'),
                     reprocess_data=False,
                     reprocess_cloud=False)
cloud.process_cloud()
```

### Running evaluation

There are a few scoring objects depending on what you want to do. The main ones are in `geometric/anchorScorer.py` and `react/reactAnchorScorer.py`.
The latter is identical but additionally filters non-clashing and interacting decorators for those that are considered feasible by AiZynthFinder.

One of the things to address is how to assess individual vector atoms. A vector atom is an atom attached to a hydrogen or terminal atom, and the evaluation is run for individual vectors and rotations of the hydrogen atoms -- I then took the average over results for each possible vector/position for the results in the thesis chapter but I think there are better ways to do this!

Example of how to run in Python (example script is also in `geometric/runScoring.py`):

```
from elaboratability.geometric.anchorScorer import AnchorScorer
from elaboratability.preprocessing.cloud import ClusterCloud

cloud = ClusterCloud(clustered_conformer_file,  # from rotations.py
                     clustered_conformer_json,  # from rotations.py
                     processed_data_file,  # if already run cloud.process_cloud()
                     False,
                     cloud_file,  # if already run cloud.process_cloud()
                     False)
cloud.process_cloud()

scorer = AnchorScorer(precursor_mol,  # rdkit mol of mol to assess
                      pdb_file,  # pdb file for protein alone
                      cloud)        
scorer.prepare_molecules()
scorer.get_vectors()
print(scorer.vectors)

scorer.evaluate_all_vectors()  # this will give the scorer.results dictionary will all the data in it
print(scorer.results)  # holds data I used for results

scorer.binary_scorer()  # I didn't use this for the results in the thesis chapter
print(scorer.scored)  # function to assign pass/fail to vectors based on some rudimentary thresholds
```

In `scorer.results` for the 'AnchorScorer' objects, the evaluation is run for every possible attached atom (and every position) to each vector atom. They are all given an ID in the dictionary `scorer.results` and the relevant atom indices (as numbered in RDKit) and the coordinates are stored.

### Preparing validation data

The code in `preprocessing/prepareValidationData.py` was used to generate validation dataset from PDBbind.
For each ligand, AiZynthFinder is used to propose precursors. The MCS between the precursor and the ligand is used to place the precursor using RDKit constrained embedding.
The relevant vector atom(s) is then recorded -- filtering is applied to precursors and vectors (as described in thesis chapter). This creates SDF files for precursors where the vector atoms are stored as a mol prop (there may be multiple precursors for a single ligand so they are named as CODE-IDX.sdf).

One thing to note is how to label vector atoms. In the code, I differentiated between vectors that are bound to a hydrogen or a terminal atom (where the hydrogen or attached atom is replaced by the elaboration, and the originating atom in the vector is labelled as the vector atom). This is useful for this method because it affects the chemical feasibility evaluation, however, for the DL method I did not differentiate between the two types. This should be considered in future iterations of the tool!
