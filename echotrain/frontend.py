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


class PasstFrontend(nn.Module):
    """Log-mel frontend that PaSST was pre-trained with (AugmentMelSTFT in hear21passt).

    Pre-emphasis, power STFT (Hann, centered frames), Kaldi mel filters, log,
    then the fixed normalisation (x + 4.5) / 5. In training mode the mel band
    edges are jittered (fmin_aug_range, fmax_aug_range) and SpecAugment masking
    is applied, as in the original.

    Input:  (batch, samples).   Output: (batch, n_mels, time).
    With the defaults, 20 s at 32 kHz gives 128 mel bins x 2001 frames.
    """

    def __init__(self, sr=32000, win_length=800, hopsize=320, n_fft=1024, n_mels=128,
                 fmin=0.0, fmax=None, freqm=48, timem=192, fmin_aug_range=10, fmax_aug_range=2000):
        super().__init__()
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
        if not fmin < fmax <= sr / 2:
            raise ValueError(f"PaSST frontend: need fmin < fmax <= sr/2, got fmin={fmin}, fmax={fmax}, sr={sr}")
        self.sr, self.n_mels, self.n_fft = sr, n_mels, n_fft
        self.win_length, self.hopsize = win_length, hopsize
        self.fmin, self.fmax = fmin, fmax
        self.fmin_aug_range, self.fmax_aug_range = max(1, int(fmin_aug_range)), max(1, int(fmax_aug_range))
        self.register_buffer("window", torch.hann_window(win_length, periodic=False), persistent=False)
        self.register_buffer("preemphasis", torch.as_tensor([[[-0.97, 1.0]]]), persistent=False)
        self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True) if freqm else nn.Identity()
        self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True) if timem else nn.Identity()

    def _mel_basis(self, fmin, fmax, device):
        basis = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels, self.n_fft, self.sr, fmin, fmax,
                                                          vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0)
        basis = basis[0] if isinstance(basis, tuple) else basis
        return nn.functional.pad(basis, (0, 1)).to(device)  # (n_mels, n_fft // 2 + 1)

    def forward(self, x):
        x = nn.functional.conv1d(x.float().unsqueeze(1), self.preemphasis).squeeze(1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                              center=True, window=self.window, return_complex=True).abs() ** 2
            if self.training:
                fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
                fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()
                fmax = min(fmax, self.sr / 2)
            else:
                fmin, fmax = self.fmin, self.fmax
            mel = torch.log(self._mel_basis(fmin, fmax, x.device) @ spec + 1e-5)  # (batch, n_mels, time)
        if self.training:
            mel = self.timem(self.freqm(mel))
        return (mel + 4.5) / 5.0
