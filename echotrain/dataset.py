"""PyTorch dataset: loads a chunk, fixes its length, applies training augmentation."""
import warnings

import librosa
import numpy as np
import torch
from torch_audiomentations import AddBackgroundNoise, AddColoredNoise, Shift


def fix_length(x, n_samples, random_crop):
    """Crop long chunks (randomly when training) and tile short chunks up to n_samples."""
    if len(x) == 0:
        return np.zeros(n_samples, dtype=np.float32)
    if len(x) > n_samples:
        start = np.random.randint(0, len(x) - n_samples + 1) if random_crop else 0
        return x[start:start + n_samples]
    if len(x) < n_samples:
        return np.tile(x, int(np.ceil(n_samples / len(x))))[:n_samples]
    return x


class AudioDataset(torch.utils.data.Dataset):
    """Returns ({"audio": waveform, "index": i}, species_target, genus_target).

    Targets are multi-hot vectors (mixup sets two classes). Without a genus head
    the genus target is an empty tensor.
    """

    def __init__(self, split, n_species, n_genus, cfg, train=False):
        self.split = split
        self.n_species = n_species
        self.n_genus = n_genus
        self.train = train
        self.sr = cfg["audio"]["sample_rate"]
        self.n_samples = int(round(cfg["audio"]["clip_seconds"] * self.sr))

        aug = cfg["augment"]
        self.mixup_rate = aug["mixup_rate"] if train else 0.0
        self.mixup_alpha = aug["mixup_alpha"]
        self.waveform_augs = []
        if train and aug["colored_noise_prob"] > 0:
            self.waveform_augs.append(AddColoredNoise(
                p=aug["colored_noise_prob"], sample_rate=self.sr, min_snr_in_db=25, max_snr_in_db=40,
                min_f_decay=-2, max_f_decay=1.5, mode="per_example", p_mode="per_example", output_type="dict"))
        if train and aug["shift_prob"] > 0:
            self.waveform_augs.append(Shift(
                p=aug["shift_prob"], sample_rate=self.sr, mode="per_example", p_mode="per_example", output_type="dict"))
        if train and aug["background_noise_prob"] > 0:
            if not aug["background_noise_dir"]:
                raise ValueError("augment.background_noise_prob > 0 needs augment.background_noise_dir")
            self.waveform_augs.append(AddBackgroundNoise(
                background_paths=aug["background_noise_dir"], p=aug["background_noise_prob"],
                sample_rate=self.sr, min_snr_in_db=3.0, max_snr_in_db=30.0, output_type="dict"))

    def __len__(self):
        return len(self.split)

    def _load(self, i):
        path = self.split.paths[i]
        try:
            # Files already at the target rate are not resampled.
            x, _ = librosa.load(path, sr=self.sr, mono=True)
        except Exception as e:  # keep training if one file is broken, but say so
            warnings.warn(f"Could not load {path}: {e}. Using silence instead.", stacklevel=2)
            x = np.zeros(self.n_samples, dtype=np.float32)
        return torch.from_numpy(fix_length(x, self.n_samples, random_crop=self.train)).float()

    def _targets(self, i):
        y = torch.zeros(self.n_species)
        y[self.split.species_idx[i]] = 1.0
        g = torch.zeros(self.n_genus)
        if self.n_genus:
            g[self.split.genus_idx[i]] = 1.0
        return y, g

    def __getitem__(self, index):
        x = self._load(index)
        y, g = self._targets(index)

        if self.mixup_rate > 0 and np.random.rand() < self.mixup_rate:
            other = np.random.randint(len(self))
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            x = lam * x + (1 - lam) * self._load(other)
            y2, g2 = self._targets(other)
            # Both sounds are present in the mix, so both classes are positive.
            y, g = torch.clamp(y + y2, max=1.0), torch.clamp(g + g2, max=1.0)

        for aug in self.waveform_augs:
            x = aug(x[None, None]).samples[0, 0]

        return {"audio": x, "index": index}, y, g
