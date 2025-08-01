import argparse
import time
import os
import glob
import torch
import librosa
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
from ml_collections import ConfigDict
from typing import Tuple, Dict, List, Union

from utils.settings import get_model_from_config, logging, write_results_in_file, parse_args_valid
from utils.audio_utils import draw_spectrogram, normalize_audio, denormalize_audio, read_audio_transposed
from utils.model_utils import demix, prefer_target_instrument, apply_tta, load_start_checkpoint
from utils.metrics import get_metrics

import warnings
import logging
from colorlog import ColoredFormatter
import math
import onnxslim
import onnx
import tarfile as tf

warnings.filterwarnings("ignore")



def setup_logging():
    """配置日志系统，同时输出到控制台和文件"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "train.log")
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 创建logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 清除现有的handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 控制台颜色格式化器
    color_formatter = ColoredFormatter(
        "%(green)s%(asctime)s%(reset)s - "
        "%(log_color)s%(levelname)s%(reset)s - %(white)s%(message)s%(reset)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={"asctime": {"green": "green"}, "name": {"blue": "blue"}},
    )
    
    # 创建文件handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(color_formatter)
    
    # 添加handler到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="Converted onnx model path")

    parser.add_argument("--model_type", "-t", type=str, default="bs_roformer", choices=["bs_roformer", "mel_band_roformer", "scnet"], help="Read README.md for reference")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random_data", action="store_true")
    parser.add_argument("--input_audio", type=str, default=None)
    parser.add_argument("--calibration_dataset", type=str, default="./calibration_dataset")
    return parser.parse_args()


def load_model(args):
    if args.model_type == "bs_roformer":
        config_file = "configs/config_bs_roformer_384_8_2_485100.yaml"
        ckpt_file = "results/model_bs_roformer_ep_17_sdr_9.6568.ckpt"
    elif args.model_type == "mel_band_roformer":
        config_file = "configs/model_mel_band_roformer_experimental_ep_53_sdr_5.1235_config_mel_64_2_1_88200_experimental.yaml"
        ckpt_file = "results/model_mel_band_roformer_experimental_ep_53_sdr_5.1235.ckpt"
    elif args.model_type == "scnet":
        config_file = "configs/config_musdb18_scnet.yaml"
        ckpt_file = "results/scnet_checkpoint_musdb18.ckpt"

    model, config = get_model_from_config(args.model_type, config_file)

    state_dict = torch.load(ckpt_file, map_location=args.device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.eval()
    return model, config


def preprocess(mix):
    # to stft
    stft_n_fft = 2048
    stft_win_length = 2048
    stft_hop_length = 441
    stft_normalized = False

    stft_kwargs = dict(
        n_fft=stft_n_fft,
        hop_length=stft_hop_length,
        win_length=stft_win_length,
        normalized=stft_normalized
    )
    device = torch.device("cpu")

    if isinstance(mix, np.ndarray):
        mix = torch.from_numpy(mix)
    b, c, l = mix.shape
    mix = mix.view(-1, l)

    stft_window = torch.hann_window(stft_win_length, device=device)

    stft_repr = torch.stft(mix, 
                           **stft_kwargs,
                           window=stft_window, 
                           return_complex=True)
    stft_repr = torch.view_as_real(stft_repr)
    # print(f"stft_repr.shape: {stft_repr.shape}")

    # stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

    # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
    # stft_repr = rearrange(stft_repr,'b s f t c -> b (f s) t c')
    s, f, t, c = stft_repr.shape
    stft_repr = stft_repr.unsqueeze(0).reshape(b, s, f, t, c).transpose(2, 1).reshape(b, -1, t, c)

    return stft_repr


def main():
    args = get_args()
    logger = setup_logging()

    model, config = load_model(args)

    instruments = config["training"]["instruments"]
    chunk_size = config["audio"]["chunk_size"]
    num_channels = config["audio"]["num_channels"]
    sample_rate = config["audio"]["sample_rate"]
    
    logger.info("-" * 70)
    logger.info(f"model_type: {args.model_type}")
    logger.info(f"onnx: {args.onnx}")
    logger.info(f"instruments: {instruments}")
    logger.info(f"chunk_size: {chunk_size}")
    logger.info(f"num_channels: {num_channels}")
    logger.info(f"sample_rate: {sample_rate}")
    logger.info(f"random_data: {args.random_data}")

    if args.random_data:
        if args.model_type == "bs_roformer":
            dim_freqs_in = config["model"]["dim_freqs_in"]
            stft_hop_length = config["model"]["stft_hop_length"]

            batch_size = 1
            time_domain_shape = (batch_size, num_channels, chunk_size)
            num_frames = int(math.ceil(chunk_size / stft_hop_length))
            # b (f s) t c
            freq_domain_shape = (batch_size, dim_freqs_in * num_channels, num_frames, 2)

            time_domain_input = torch.randn(time_domain_shape)
            freq_domain_input = torch.randn(freq_domain_shape)

            model.forward = model.forward_for_export
            mask, stft_output = model(freq_domain_input)
            # stft_output = model(freq_domain_input)
            print(f"mask.shape: {mask.shape}")
            print(f"stft_output.shape: {stft_output.shape}")

            input_names = ["stft_input",]
            output_names = ["mask", "stft_output",]

            inputs = (
                freq_domain_input,
            )

            os.makedirs(args.calibration_dataset, exist_ok=True)
            for i, name in enumerate(input_names):
                with tf.open(os.path.join(args.calibration_dataset, name + ".tar.gz"), "w:gz") as f:
                    np.save(os.path.join(args.calibration_dataset, name + ".npy"), inputs[i].numpy())
                    f.add(os.path.join(args.calibration_dataset, name + ".npy"))
        

            onnx_dir = os.path.dirname(args.onnx)
            if onnx_dir != '':
                os.makedirs(onnx_dir, exist_ok=True)

            logger.info("Exporting model to onnx...")
            with torch.no_grad():
                torch.onnx.export(model,               # model being run
                    inputs,                    # model input (or a tuple for multiple inputs)
                    args.onnx,              # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=16,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=input_names, # the model's input names
                    output_names=output_names, # the model's output names
                    dynamic_axes=None,
                    optimize=True
                )
                slim_model = onnxslim.slim(args.onnx)
                onnx.save(slim_model, args.onnx)
                logger.info(f"Successfully export onnx to {args.onnx}")
        elif args.model_type == "mel_band_roformer":
            assert os.path.exists(args.input_audio)
            wav, origin_sr = sf.read(args.input_audio, always_2d=True, dtype="float32")
            if origin_sr != sample_rate:
                print(f"Origin sample rate is {origin_sr}, resampling to {sample_rate}...")
                wav = librosa.resample(wav, orig_sr=origin_sr, target_sr=sample_rate)
            if wav.shape[0] != 2:
                wav = wav.transpose()

            ref = wav.mean(0)
            wav -= ref.mean()
            wav /= ref.std() + 1e-8
            mix = wav[np.newaxis, ...]

            dim_freqs_in = config["model"]["dim_freqs_in"]
            stft_hop_length = config["model"]["stft_hop_length"]

            batch_size = 1
            time_domain_shape = (batch_size, num_channels, chunk_size)
            num_frames = int(math.ceil(chunk_size / stft_hop_length)) + 1
            # b (f s) t c
            
            # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
            # stft_repr = rearrange(stft_repr,'b s f t c -> b (f s) t c')
            freq_domain_shape = (batch_size, dim_freqs_in * num_channels, num_frames, 2)

            time_domain_input = torch.randn(time_domain_shape)
            # freq_domain_input = torch.randn(freq_domain_shape, device=args.device)
            freq_domain_input = preprocess(mix[..., chunk_size*10:chunk_size*10 + chunk_size])

            # batch_arange = torch.arange(batch_size, device=args.device)[..., None]
            # freq_domain_input = freq_domain_input[batch_arange, model.freq_indices]
            print(f"freq_domain_input: {freq_domain_input.shape}")

            model.forward = model.forward_for_export
            masks = model(freq_domain_input)
            # masks = model(time_domain_input)
            print(f"mask.shape: {masks.shape}")

            input_names = ["stft_input",]
            output_names = ["masks",]

            inputs = (
                freq_domain_input,
            )

            os.makedirs(args.calibration_dataset, exist_ok=True)
            for i, name in enumerate(input_names):
                with tf.open(os.path.join(args.calibration_dataset, name + ".tar.gz"), "w:gz") as f:
                    np.save(os.path.join(args.calibration_dataset, name + ".npy"), inputs[i].numpy())
                    f.add(os.path.join(args.calibration_dataset, name + ".npy"))
        

            onnx_dir = os.path.dirname(args.onnx)
            if onnx_dir != '':
                os.makedirs(onnx_dir, exist_ok=True)

            logger.info("Exporting model to onnx...")
            with torch.no_grad():
                torch.onnx.export(model,               # model being run
                    inputs,                    # model input (or a tuple for multiple inputs)
                    args.onnx,              # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=16,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=input_names, # the model's input names
                    output_names=output_names, # the model's output names
                    dynamic_axes=None,
                    optimize=True
                )
                slim_model = onnxslim.slim(args.onnx)
                onnx.save(slim_model, args.onnx)
                logger.info(f"Successfully export onnx to {args.onnx}")
        elif args.model_type == "scnet":
            nfft = config["model"]["nfft"]
            hop_size = config["model"]["hop_size"]

            batch_size = 1
            # time_domain_shape = (batch_size, num_channels, chunk_size)
            num_frames = int(math.ceil(chunk_size / hop_size))
            # b (f s) t c
            freq_domain_shape = (batch_size, 2 * num_channels, nfft // 2 + 1, num_frames)

            # time_domain_input = torch.randn(time_domain_shape)
            freq_domain_input = torch.randn(freq_domain_shape)

            model.forward = model.forward_for_export
            stft_output = model(freq_domain_input)
            # stft_output = model(time_domain_input)
            print(f"stft_output.shape: {stft_output.shape}")

            input_names = ["stft_input",]
            output_names = ["stft_output",]

            inputs = (
                freq_domain_input,
            )

            os.makedirs(args.calibration_dataset, exist_ok=True)
            for i, name in enumerate(input_names):
                with tf.open(os.path.join(args.calibration_dataset, name + ".tar.gz"), "w:gz") as f:
                    np.save(os.path.join(args.calibration_dataset, name + ".npy"), inputs[i].numpy())
                    f.add(os.path.join(args.calibration_dataset, name + ".npy"))
        

            onnx_dir = os.path.dirname(args.onnx)
            if onnx_dir != '':
                os.makedirs(onnx_dir, exist_ok=True)

            logger.info("Exporting model to onnx...")
            with torch.no_grad():
                torch.onnx.export(model,               # model being run
                    inputs,                    # model input (or a tuple for multiple inputs)
                    args.onnx,              # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=16,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=input_names, # the model's input names
                    output_names=output_names, # the model's output names
                    dynamic_axes=None,
                    optimize=True
                )
                slim_model = onnxslim.slim(args.onnx)
                onnx.save(slim_model, args.onnx)
                logger.info(f"Successfully export onnx to {args.onnx}")
    else:
        logger.error(f"Unknown model type: {args.model_type}")

if __name__ == "__main__":
    main()
