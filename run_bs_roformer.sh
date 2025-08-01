#!/bin/bash

conda activate msst
python inference.py --model_type bs_roformer --config_path configs/config_bs_roformer_384_8_2_485100.yaml --start_check_point results/model_bs_roformer_ep_17_sdr_9.6568.ckpt --input_folder input/wavs/ --store_dir separation_results/
conda deactivate
