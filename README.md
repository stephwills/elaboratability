# Elaboratability evaluator

This tool is under development! A brute-force approach to evaluating the 'elaboratability' of a compound.

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
