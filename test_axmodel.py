from axengine import InferenceSession
import numpy as np
import argparse
import os
import soundfile as sf
import librosa
import torch
import time
import glob
import tqdm
from librosa import filters
import onnxruntime as ort
from einops import rearrange, pack, unpack, reduce, repeat


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio", "-i", type=str, required=True, help="Input audio file(.wav)")
    parser.add_argument("--output_path", "-o", type=str, required=False, default="./output", help="Seperated wav path")
    parser.add_argument("--model", "-m", type=str, required=False, default="./mel_band_roformer.axmodel", help="demucs onnx model")
    parser.add_argument("--overlap", type=float, required=False, default=0.25)
    parser.add_argument("--segment", type=float, required=False, default=2, help="Split in seconds")
    parser.add_argument("--vocals_only", action="store_true")
    return parser.parse_args()


def calc_freq_indices():
    stft_n_fft = 2048
    stft_win_length = 2048
    stft_hop_length = 441
    stft_normalized = False
    sample_rate = 44100
    num_bands = 60
    stereo = True

    stft_kwargs = dict(
        n_fft=stft_n_fft,
        hop_length=stft_hop_length,
        win_length=stft_win_length,
        normalized=stft_normalized
    )

    freqs = torch.stft(torch.randn(1, 4096), **stft_kwargs, window=torch.ones(stft_n_fft), return_complex=True).shape[1]

    # create mel filter bank
    # with librosa.filters.mel as in section 2 of paper

    mel_filter_bank_numpy = filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)

    mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

    # for some reason, it doesn't include the first freq? just force a value for now

    mel_filter_bank[0][0] = 1.

    # In some systems/envs we get 0.0 instead of ~1.9e-18 in the last position,
    # so let's force a positive value

    mel_filter_bank[-1, -1] = 1.

    # binary as in paper (then estimated masks are averaged for overlapping regions)

    freqs_per_band = mel_filter_bank > 0
    assert freqs_per_band.any(dim=0).all(), 'all frequencies need to be covered by all bands for now'

    repeated_freq_indices = repeat(torch.arange(freqs), 'f -> b f', b=num_bands)
    freq_indices = repeated_freq_indices[freqs_per_band]

    if stereo:
        freq_indices = repeat(freq_indices, 'f -> f s', s=2)
        freq_indices = freq_indices * 2 + torch.arange(2)
        freq_indices = rearrange(freq_indices, 'f s -> (f s)')

    num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
    num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')

    return freq_indices, freqs_per_band, num_freqs_per_band, num_bands_per_freq


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

    return stft_repr.numpy()


def postprocess(masks, stft_repr, freq_indices, num_bands_per_freq, audio_len, num_stems=4, channels=2):
    masks = torch.from_numpy(masks)
    stft_repr = torch.from_numpy(stft_repr)
    batch = 1
    stft_n_fft = 2048
    stft_win_length = 2048
    stft_hop_length = 441
    stft_normalized = False
    istft_length = audio_len

    stft_kwargs = dict(
        n_fft=stft_n_fft,
        hop_length=stft_hop_length,
        win_length=stft_win_length,
        normalized=stft_normalized
    )

    device = torch.device("cpu")
    stft_window = torch.hann_window(stft_win_length, device=device)

    # modulate frequency representation

    stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

    # complex number multiplication

    stft_repr = torch.view_as_complex(stft_repr)
    masks = torch.view_as_complex(masks)

    masks = masks.type(stft_repr.dtype)

    # need to average the estimated mask for the overlapped frequencies

    scatter_indices = repeat(freq_indices, 'f -> b n f t', b=1, n=num_stems, t=stft_repr.shape[-1])

    stft_repr_expanded_stems = repeat(stft_repr, 'b 1 ... -> b n ...', n=num_stems)
    masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(2, scatter_indices, masks)

    denom = repeat(num_bands_per_freq, 'f -> (f r) 1', r=channels)
    # print(f"stft_repr.shape: {stft_repr.shape}")
    # print(f"stft_repr_expanded_stems.shape: {stft_repr_expanded_stems.shape}")
    # print(f"masks_summed.shape: {masks_summed.shape}")
    # print(f"denom.shape: {denom.shape}")

    masks_averaged = masks_summed / denom.clamp(min=1e-8)

    # modulate stft repr with estimated mask

    stft_repr = stft_repr * masks_averaged

    # istft

    stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=2)

    recon_audio = torch.istft(stft_repr, **stft_kwargs, window=stft_window, return_complex=False,
                              length=istft_length)

    recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch, s=2, n=num_stems)

    if num_stems == 1:
        recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

    return recon_audio.numpy()


def apply_model(model,
                mix,
                freq_indices,
                num_bands_per_freq,
                segment,
                overlap: float = 0.25,
                len_model_sources=4,
                samplerate=44100
                ):
    """
    :param mix:
    :param overlap:
    :param device:
    :param transition_power:
    :param len_model_sources:
    :param segment:
    :param samplerate:
    :param model:
    :return:
    """
    model_weights = [1.]*len_model_sources
    totals = [0.] * len_model_sources
    batch, channels, length = mix.shape

    segment_length: int = int(samplerate * segment)
    stride = int((1 - overlap) * segment_length)
    futures = []

    for offset in tqdm.tqdm(range(0, length, stride)):
        chunk = mix[..., offset:offset + segment_length]
        audio_len = chunk.shape[-1]
        if chunk.shape[-1] < segment_length:
            chunk = np.concatenate([chunk, np.zeros((batch, channels, segment_length - chunk.shape[-1]), dtype=np.float32)], axis=-1)

        stft_input = preprocess(chunk)
        masks = model.run(None, {"stft_input": stft_input})[0]
        future = postprocess(masks, stft_input, freq_indices, num_bands_per_freq, audio_len, num_stems=len_model_sources)
        future = future[..., :audio_len]

        futures.append((future, offset))
        # offset += segment_length

    out = np.zeros((batch, len_model_sources, channels, length))
    sum_weight = np.zeros((length,))
    weight = np.concatenate([np.arange(1, segment_length // 2 + 1),
                        np.arange(segment_length - segment_length // 2, 0, -1)], axis=-1)
    weight = weight / weight.max()
    for future, offset in futures:
        chunk_out = future
        chunk_length = chunk_out.shape[-1]
        out[..., offset:offset + segment_length] += (weight[:chunk_length] * chunk_out)
        sum_weight[offset:offset + segment_length] += weight[:chunk_length]
    out /= sum_weight

    for k, inst_weight in enumerate(model_weights):
        out[:, k, :, :] *= inst_weight
        totals[k] += inst_weight
    for k in range(out.shape[1]):
        out[:, k, :, :] /= totals[k]
    return out


def main():
    args = get_args()
    assert os.path.exists(args.input_audio), f"Input audio {args.input_audio} not exist"
    assert os.path.exists(args.model), f"Model {args.model} not exist"
    os.makedirs(args.output_path, exist_ok=True)

    input_audio = args.input_audio
    output_path = args.output_path
    model_path = args.model
    segment = args.segment
    num_stems = 1 if args.vocals_only else 4

    target_sr = 44100

    print(f"Input audio: {input_audio}")
    print(f"Output path: {output_path}")
    print(f"Model: {model_path}")
    print(f"Overlap: {args.overlap}")

    if os.path.isdir(input_audio):
        types = ('*.wav', '*.mp3', '*.flac') # the tuple of file types
        input_audios = []
        for files in types:
            input_audios.extend(glob.glob(f"{input_audio}/**/{files}", recursive=True))
    else:
        input_audios = [input_audio]

    freq_indices, freqs_per_band, num_freqs_per_band, num_bands_per_freq = calc_freq_indices()

    for input_audio in input_audios:
        print(f"Loading audio {input_audio}...")
        wav, origin_sr = sf.read(input_audio, always_2d=True, dtype="float32")
        if origin_sr != target_sr:
            print(f"Origin sample rate is {origin_sr}, resampling to {target_sr}...")
            wav = librosa.resample(wav, orig_sr=origin_sr, target_sr=target_sr)
        if wav.shape[0] != 2:
            wav = wav.transpose()
        # print(wav.shape)

        print("Loading model...")
        start = time.time()
        if os.path.splitext(model_path)[1] == ".axmodel":
            sess = InferenceSession(model_path, providers=['AxEngineExecutionProvider'])
        else:
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print(f"Load model take {1000 * (time.time() - start)}ms")

        print("Preprocessing audio...")
        start = time.time()
        ref = wav.mean(0)
        ref_mean = ref.mean()
        ref_std = ref.std()
        preprocessed_wav = (wav - ref_mean) / (ref_std + 1e-8)
        # wav = torch.from_numpy(wav)
        print(f"preprocess audio take {1000 * (time.time() - start)}ms")

        print("Running model...")
        out = apply_model(
            sess,
            preprocessed_wav[None],
            freq_indices,
            num_bands_per_freq,
            segment=segment,
            overlap=args.overlap,
            len_model_sources=num_stems
        )

        print("Postprocessing...")
        out *= ref_std + 1e-8
        out += ref_mean
        # wav *= ref.std() + 1e-8
        # wav += ref.mean()

        # out = out.numpy()

        if args.vocals_only:
            out = out[0]
            vocals = out[0]
            other = wav - vocals

            sources = ['vocals', 'other']
            res = dict(zip(sources, [vocals, other]))
            print("Saving audio...")
            for name, source in res.items():
                source = source / max(1.01 * np.abs(source).max(), 1)
                
                if source.shape[1] != 2:
                    source = source.transpose()

                audio_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(input_audio))[0]}_{name}.wav")
                sf.write(audio_path, source, samplerate=target_sr)
                print(f"Save {name} to {audio_path}")
        else:
            sources = ['drums', 'bass', 'other', 'vocals']
            res = dict(zip(sources, out[0]))
            print("Saving audio...")
            for name, source in res.items():
                source = source / max(1.01 * np.abs(source).max(), 1)
                
                if source.shape[1] != 2:
                    source = source.transpose()

                audio_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(input_audio))[0]}_{name}.wav")
                sf.write(audio_path, source, samplerate=target_sr)
                print(f"Save {name} to {audio_path}")


if __name__ == "__main__":
    main()
