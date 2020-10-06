#!/bin/bash

# Refer to https://github.com/chnsh/DCRNN_PyTorch. Download and put the h5 datasets into data/ folder.

mkdir -p data/{METR-LA,PEMS-BAY}
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5