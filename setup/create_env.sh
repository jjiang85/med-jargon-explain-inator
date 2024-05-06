#!/bin/sh
# Example: source ~/miniconda3/etc/profile.d/conda.sh
source $1
conda remove -n MedJarg --all
conda env create -f requirements.yml
conda activate MedJarg
python -m nltk.downloader all