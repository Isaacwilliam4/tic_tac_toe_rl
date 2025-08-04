#!/bin/bash

# Name of the environment
ENV_NAME="tic_tac_rl"

# Create the conda environment with Python 3.10
conda create -y -n $ENV_NAME python=3.10

# Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install required packages
conda install -y numpy pytest

# Optional: install editable local version of your package
pip install -e .

echo "Conda environment '$ENV_NAME' created and package installed."
