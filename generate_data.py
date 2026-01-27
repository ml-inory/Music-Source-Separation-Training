import numpy as np
import argparse
import os
import glob
import librosa
import tqdm
import tarfile as tf
from mel_band_roformer import MelBandRoformer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio", "-i", type=str, required=True, help="Input audio file(.wav)")
    parser.add_argument("--calib_path", "-o", type=str, required=False, default="./calib_dataset", help="Seperated wav path")
    parser.add_argument("--model", "-m", type=str, required=False, default="./mel_band_roformer.axmodel", help="demucs onnx model")
    parser.add_argument("--overlap", type=float, required=False, default=0.25)
    parser.add_argument("--num_stems", type=int, default=4)
    parser.add_argument("--segment", type=int, required=False, default=88200, help="Split in seconds")
    return parser.parse_args()


def main():
    args = get_args()
    assert os.path.exists(args.input_audio), f"Input audio {args.input_audio} not exist"
    assert os.path.exists(args.model), f"Model {args.model} not exist"
    os.makedirs(args.calib_path, exist_ok=True)

    input_audio = args.input_audio
    calib_path = args.calib_path
    model_path = args.model
    segment = args.segment
    num_stems = args.num_stems
    sample_rate = 44100

    print(f"Input audio: {input_audio}")
    print(f"Calib path: {calib_path}")
    print(f"Model: {model_path}")
    print(f"Segment: {segment}")
    print(f"num_stems: {num_stems}")
    print(f"Overlap: {args.overlap}")

    model = MelBandRoformer(
        model_path,
        input_len=segment,
        num_stems=num_stems
    )

    if os.path.isdir(input_audio):
        types = ('*.wav', '*.mp3', '*.flac') # the tuple of file types
        input_audios = []
        for files in types:
            input_audios.extend(glob.glob(f"{input_audio}/**/{files}", recursive=True))
    else:
        input_audios = [input_audio]

    tf_file = tf.open(os.path.join(calib_path, "stft_input.tar.gz"), "w:gz")

    for input_audio in input_audios:
        audio_name = os.path.splitext(os.path.basename(input_audio))[0]
        os.makedirs(os.path.join(calib_path, audio_name), exist_ok=True)

        wav, _ = librosa.load(input_audio, sr=sample_rate, mono=False)
        if wav.shape[0] != 2:
            wav = wav.transpose()

        ref = wav.mean(0)
        ref_mean = ref.mean()
        ref_std = ref.std()
        preprocessed_wav = (wav - ref_mean) / (ref_std + 1e-8)
        mix = preprocessed_wav[np.newaxis, ...]

        batch, channels, length = mix.shape

        stride = int((1 - args.overlap) * segment)

        data_index = 0
        for offset in tqdm.tqdm(range(0, length, stride)):
            chunk = mix[..., offset:offset + segment]
            if chunk.shape[-1] < segment:
                chunk = np.concatenate([chunk, np.zeros((batch, channels, segment - chunk.shape[-1]), dtype=np.float32)], axis=-1)

            stft_input = model.preprocess(chunk)

            npy_path = os.path.join(calib_path, audio_name, f"stft_input_{data_index}.npy")
            np.save(npy_path, stft_input)
            tf_file.add(npy_path)

            data_index += 1

    tf_file.close()

if __name__ == "__main__":
    main()
