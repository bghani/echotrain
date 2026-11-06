"""Waveform -> log-mel spectrogram, computed on the GPU inside the model."""
import torch
import torch.nn as nn
import torchaudio


class PerchFrontend(nn.Module):
    """Log-mel frontend following the Perch 2.0 spec (arXiv 2508.04665, App. A.2).

    Magnitude (not power) STFT, Hann window, uncentered frames, HTK mel scale,
    scaled by 1/sum(window), then 0.1 * log(mel + 1e-5). SpecAugment
    (frequency and time masking) is applied in training mode only.

    Input:  (batch, samples).   Output: (batch, time, n_mels).
    With the defaults, 5 s at 32 kHz gives 500 frames x 128 mel bins.
    """

    def __init__(self, sr=32000, win_length=640, hopsize=320, n_fft=1024, n_mels=128,
                 fmin=60.0, fmax=16000.0, log_floor=1e-5, log_scalar=0.1, freqm=48, timem=192):
        super().__init__()
        self.hopsize, self.n_fft, self.win_length = hopsize, n_fft, win_length
        self.log_floor, self.log_scalar = log_floor, log_scalar

        window = torch.hann_window(win_length, periodic=True)
        self.register_buffer("window", window, persistent=False)
        self.register_buffer("window_sum", window.sum(), persistent=False)
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1, f_min=fmin, f_max=fmax, n_mels=n_mels,
            sample_rate=sr, norm=None, mel_scale="htk")
        self.register_buffer("mel_fb", mel_fb, persistent=False)

        self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True) if freqm else nn.Identity()
        self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True) if timem else nn.Identity()

    def forward(self, x):
        # Pad so that the number of frames is ceil(samples / hop) with center=False.
        frames = -(-x.shape[-1] // self.hopsize)
        pad = max((frames - 1) * self.hopsize + self.n_fft - x.shape[-1], 0)
        x = nn.functional.pad(x, (0, pad))

        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                          window=self.window, center=False, return_complex=True)
        mag = torch.sqrt(stft.real ** 2 + stft.imag ** 2 + 1e-12) / self.window_sum
        mel = mag.transpose(1, 2) @ self.mel_fb                 # (batch, time, n_mels)
        out = torch.log(mel + self.log_floor) * self.log_scalar

        if self.training:
            out = out.transpose(1, 2)                           # maskers expect (..., freq, time)
            out = self.timem(self.freqm(out)).transpose(1, 2)
        return out
