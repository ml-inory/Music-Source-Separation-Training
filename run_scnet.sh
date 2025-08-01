#!/bin/bash

conda activate msst
python inference.py --model_type scnet --config_path configs/config_musdb18_scnet.yaml --start_check_point results/scnet_checkpoint_musdb18.ckpt --input_folder input/wavs/ --store_dir separation_results/
conda deactivate
