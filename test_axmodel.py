import numpy as np
import argparse
import os
import soundfile as sf
import glob
from mel_band_roformer import MelBandRoformer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio", "-i", type=str, required=True, help="Input audio file(.wav)")
    parser.add_argument("--output_path", "-o", type=str, required=False, default="./output", help="Seperated wav path")
    parser.add_argument("--model", "-m", type=str, required=False, default="./mel_band_roformer.axmodel", help="demucs onnx model")
    parser.add_argument("--overlap", type=float, required=False, default=0.25)
    parser.add_argument("--segment", type=int, required=False, default=88200, help="Split in seconds")
    parser.add_argument("--num_stems", type=int, default=4)
    parser.add_argument('--stem_names', metavar='STEM_NAMES', type=str, default=None, nargs='+',
                    help='output instrument names, a list of strings (space-separated)')
    return parser.parse_args()


def main():
    args = get_args()
    assert os.path.exists(args.input_audio), f"Input audio {args.input_audio} not exist"
    assert os.path.exists(args.model), f"Model {args.model} not exist"
    os.makedirs(args.output_path, exist_ok=True)

    input_audio = args.input_audio
    output_path = args.output_path
    model_path = args.model
    segment = args.segment
    num_stems = args.num_stems
    stem_names = args.stem_names

    if stem_names:
        assert num_stems == len(stem_names)

    sample_rate = 44100

    print(f"Input audio: {input_audio}")
    print(f"Output path: {output_path}")
    print(f"Model: {model_path}")
    print(f"Segment: {segment}")
    print(f"num_stems: {num_stems}")
    print(f"Overlap: {args.overlap}")

    model = MelBandRoformer(
        model_path,
        input_len=segment,
        num_stems=num_stems,
        stem_names=stem_names,
        sample_rate=sample_rate
    )

    if os.path.isdir(input_audio):
        types = ('*.wav', '*.mp3', '*.flac') # the tuple of file types
        input_audios = []
        for files in types:
            input_audios.extend(glob.glob(f"{input_audio}/**/{files}", recursive=True))
    else:
        input_audios = [input_audio]

    for input_audio in input_audios:
        output = model.run(input_audio, args.overlap)
        for name, source in output.items():
            source = source / max(1.01 * np.abs(source).max(), 1)
            
            if source.shape[1] != 2:
                source = source.transpose()

            audio_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(input_audio))[0]}_{name}.wav")
            sf.write(audio_path, source, samplerate=sample_rate)
            print(f"Save {name} to {audio_path}")

if __name__ == "__main__":
    main()
