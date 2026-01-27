import numpy as np
import os
import librosa
import torch
import tqdm
from librosa import filters
from einops import rearrange, pack, unpack, reduce, repeat
from typing import Optional, List

class MelBandRoformer:
    def __init__(self,
                 model_path,
                 input_len = 88200,
                 num_stems = 4,
                 stem_names: Optional[List[str]] = None,
                 stft_n_fft = 2048,
                 stft_win_length = 2048,
                 stft_hop_length = 441,
                 stft_normalized = False,
                 sample_rate = 44100,
                 num_bands = 60,
                 stereo = True):
        self.input_len = input_len
        self.num_stems = num_stems
        self.stem_names = stem_names
        if isinstance(stem_names, list):
            assert len(self.stem_names) == self.num_stems

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )
        self.sample_rate = sample_rate
        self.num_bands = num_bands
        self.stereo = stereo
        self.device = torch.device("cpu")

        model_format = os.path.splitext(model_path)[1]
        if model_format == '.axmodel':
            import axengine as axe
            self.model = axe.InferenceSession(model_path)
        elif model_format == '.onnx':
            import onnxruntime as ort
            self.model = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            # [batch, freq, frames, channels]
            input_shape = self.model.get_inputs()[0].shape
            assert input_len == (input_shape[2] - 1) * stft_hop_length
        else:
            raise RuntimeError(f"Unknown model format: {model_format}")

        self.freq_indices, self.freqs_per_band, self.num_freqs_per_band, self.num_bands_per_freq = \
            self.calc_freq_indices()
        self.stft_window = torch.hann_window(stft_win_length, device=self.device)


    def run(self, audio_path: str, overlap: float = 0.25) -> dict[str, np.ndarray]:
        wav, _ = librosa.load(audio_path, sr=self.sample_rate, mono=False)
        if wav.shape[0] != 2:
            wav = wav.transpose()

        ref = wav.mean(0)
        ref_mean = ref.mean()
        ref_std = ref.std()
        preprocessed_wav = (wav - ref_mean) / (ref_std + 1e-8)

        out = self.apply_model(
            preprocessed_wav[None],
            overlap=overlap
        )

        out *= ref_std + 1e-8
        out += ref_mean
        
        # [1, num_stems, channels, output_len]
        out = out[0]
        if self.stem_names is not None:
            output_names = self.stem_names
        else:
            output_names = [f"stem_{i}" for i in range(self.num_stems)]

        result = {}
        for i in range(self.num_stems):
            result[output_names[i]] = out[i]

        return result


    def calc_freq_indices(self):
        n_fft = self.stft_kwargs['n_fft']
        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(n_fft), return_complex=True).shape[1]

        # create mel filter bank
        # with librosa.filters.mel as in section 2 of paper

        mel_filter_bank_numpy = filters.mel(sr=self.sample_rate, n_fft=n_fft, n_mels=self.num_bands)

        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        # for some reason, it doesn't include the first freq? just force a value for now

        mel_filter_bank[0][0] = 1.

        # In some systems/envs we get 0.0 instead of ~1.9e-18 in the last position,
        # so let's force a positive value

        mel_filter_bank[-1, -1] = 1.

        # binary as in paper (then estimated masks are averaged for overlapping regions)

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), 'all frequencies need to be covered by all bands for now'

        repeated_freq_indices = repeat(torch.arange(freqs), 'f -> b f', b=self.num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if self.stereo:
            freq_indices = repeat(freq_indices, 'f -> f s', s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, 'f s -> (f s)')

        num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
        num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')

        return freq_indices, freqs_per_band, num_freqs_per_band, num_bands_per_freq
    

    def preprocess(self, mix) -> np.ndarray:
        if isinstance(mix, np.ndarray):
            mix = torch.from_numpy(mix)
        b, c, l = mix.shape
        mix = mix.view(-1, l)

        stft_repr = torch.stft(mix, 
                            **self.stft_kwargs,
                            window=self.stft_window, 
                            return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)

        # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
        # stft_repr = rearrange(stft_repr,'b s f t c -> b (f s) t c')
        s, f, t, c = stft_repr.shape
        stft_repr = stft_repr.unsqueeze(0).reshape(b, s, f, t, c).transpose(2, 1).reshape(b, -1, t, c)

        return stft_repr.numpy()
    

    def postprocess(self, masks, stft_repr, audio_len):
        masks = torch.from_numpy(masks)
        stft_repr = torch.from_numpy(stft_repr)
        batch = 1
        istft_length = audio_len
        channels = 2 if self.stereo else 1

        # modulate frequency representation

        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        masks = torch.view_as_complex(masks)

        masks = masks.type(stft_repr.dtype)

        # need to average the estimated mask for the overlapped frequencies

        scatter_indices = repeat(self.freq_indices, 'f -> b n f t', b=1, n=self.num_stems, t=stft_repr.shape[-1])

        stft_repr_expanded_stems = repeat(stft_repr, 'b 1 ... -> b n ...', n=self.num_stems)
        masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(2, scatter_indices, masks)

        denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=channels)

        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        # modulate stft repr with estimated mask
        stft_repr = stft_repr * masks_averaged

        # istft

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=2)

        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=self.stft_window, return_complex=False,
                                length=istft_length)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch, s=2, n=self.num_stems)

        if self.num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        return recon_audio.numpy()
    

    def apply_model(self, mix, overlap: float = 0.25):
        model_weights = [1.]*self.num_stems
        totals = [0.] * self.num_stems
        batch, channels, length = mix.shape

        stride = int((1 - overlap) * self.input_len)
        futures = []

        for offset in tqdm.tqdm(range(0, length, stride)):
            chunk = mix[..., offset:offset + self.input_len]
            audio_len = chunk.shape[-1]
            if chunk.shape[-1] < self.input_len:
                chunk = np.concatenate([chunk, np.zeros((batch, channels, self.input_len - chunk.shape[-1]), dtype=np.float32)], axis=-1)

            stft_input = self.preprocess(chunk)
            masks = self.model.run(None, {"stft_input": stft_input})[0]
            future = self.postprocess(masks, stft_input, audio_len)
            future = future[..., :audio_len]

            futures.append((future, offset))

        out = np.zeros((batch, self.num_stems, channels, length))
        sum_weight = np.zeros((length,))
        weight = np.concatenate([np.arange(1, self.input_len // 2 + 1),
                            np.arange(self.input_len - self.input_len // 2, 0, -1)], axis=-1)
        weight = weight / weight.max()
        for future, offset in futures:
            chunk_out = future
            chunk_length = chunk_out.shape[-1]
            out[..., offset:offset + self.input_len] += (weight[:chunk_length] * chunk_out)
            sum_weight[offset:offset + self.input_len] += weight[:chunk_length]
        out /= sum_weight

        for k, inst_weight in enumerate(model_weights):
            out[:, k, :, :] *= inst_weight
            totals[k] += inst_weight
        for k in range(out.shape[1]):
            out[:, k, :, :] /= totals[k]
        return out