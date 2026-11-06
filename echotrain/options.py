"""Load a YAML config, fill in defaults and apply command-line overrides."""
import copy
from pathlib import Path

import yaml

# Every option with its default value. A config file only needs to list the
# values that differ from these.
DEFAULTS = {
    "run_name": "run",
    "output_dir": "runs",
    "seed": 78,
    "device": "cuda:0",
    "data": {
        # "folders": train_dir/<species>/*.wav and val_dir/<species>/*.wav
        # "manifest": one CSV with a row per audio file (path, label, split)
        "mode": "folders",
        # folders mode
        "train_dir": None,
        "val_dir": None,
        "audio_extensions": [".wav", ".flac", ".mp3", ".ogg"],
        # manifest mode
        "manifest": None,
        "audio_root": None,      # path_col values are relative to this (absolute paths are used as-is)
        "path_col": "chunk_path",
        "label_col": "latin",
        "split_col": "split",
        "group_col": None,       # column naming the source recording of each chunk (see eval.group_regex)
        "train_splits": ["train"],
        "val_splits": ["val"],
        # both modes
        "species_list": None,    # optional text file, one class per line; restricts and fixes the class set
    },
    "audio": {
        "sample_rate": 32000,
        "clip_seconds": 20.0,    # every clip is cropped or tiled to this length
    },
    "taxonomy": {
        "genus_head": False,     # add a second output head for genus (or any higher rank)
        "genus_map": None,       # optional CSV with columns species,genus; otherwise genus = first word of the name
    },
    "model": {
        "name": "effnet_chunked", # see README section 4 / MODELS in echotrain/model_zoo.py
        "backbone": "efficientnet_b3",  # effnet_* models: any timm EfficientNet, e.g. efficientnet_b0 ... b5,
                                        # tf_efficientnet_b0 ... b8, tf_efficientnetv2_s
        "pretrained": True,      # ImageNet weights (EfficientNet) / AudioSet weights (PaSST)
        "perch_weights": None,   # backbone weights for the perch_* models
        "chunk_seconds": 5.0,    # effnet_chunked: each clip is cut into chunks of this length
        "chunk_aggregation": "max",
        "init_checkpoint": None, # warm start: load matching weights from an earlier run
        # passt / passt_single. PaSST has its own spectrogram frontend (the one it was
        # pre-trained with); of the frontend section only n_mels (keep 128), freqm and timem apply.
        "passt_arch": "passt_20sec",  # passt_20sec, passt_30sec, passt_s_kd_p16_128_ap486 (10 s), ...
        "passt_fmin": 100.0,
        "passt_fmax": None,      # None = sample_rate / 2 - 1000 Hz (the hear21passt default)
        "passt_patchout_t": 0,   # structured patchout: time patches dropped per clip in training
        "passt_patchout_f": 0,   # structured patchout: frequency patches dropped per clip in training
    },
    "frontend": {
        "fmin": 60.0,
        "fmax": 16000.0,
        "n_mels": 128,
        "freqm": 48,             # SpecAugment frequency mask (training only, 0 = off)
        "timem": 192,            # SpecAugment time mask (training only, 0 = off)
    },
    "augment": {
        "mixup_rate": 0.1,
        "mixup_alpha": 10.0,
        "colored_noise_prob": 0.0,
        "shift_prob": 0.0,
        "background_noise_prob": 0.0,
        "background_noise_dir": None,
    },
    "sampler": {
        "balanced": True,        # oversample rare classes with a weighted sampler
        "cap": 1000,             # class counts above this are treated as equal to it
        "power": 0.5,            # weight = 1 / count**power (0 = no balancing, 1 = full balancing)
        "noise_class": None,     # name of a background/noise class to down-weight
        "noise_multiplier": 0.4,
    },
    "train": {
        "epochs": 200,
        "batch_size": 16,
        "num_workers": 8,
        "lr": 1.0e-4,
        "weight_decay": 1.0e-4,
        "grad_accum_steps": 1,
        "grad_clip": 1.0,
        "amp": True,
        "genus_loss_weight": 1.0,
        "scheduler": "plateau",  # "plateau" (on validation mAP) or "none"
        "plateau_factor": 0.1,
        "plateau_patience": 4,
        "min_lr": 1.0e-6,
        "early_stop_patience": None,  # stop after this many epochs without a better mAP
    },
    "eval": {
        "batch_size": 16,
        "num_workers": 8,
        "threshold": 0.5,
        # Chunks of one recording are max-pooled into a single prediction before
        # scoring. Without data.group_col, the recording is the file name with
        # this pattern removed (e.g. "XC123_chunk004" -> "XC123").
        "group_regex": r"_(chunk)?\d+$",
        "top_n_confusions": 20,
    },
}

FLOAT_KEYS = {
    ("train", "lr"), ("train", "weight_decay"), ("train", "min_lr"),
    ("train", "plateau_factor"), ("train", "genus_loss_weight"),
}


def _merge(base, override, path=()):
    for key, value in override.items():
        if key not in base:
            raise KeyError(f"Unknown config option: {'.'.join(path + (key,))}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"Config option {'.'.join(path + (key,))} must be a section")
            _merge(base[key], value, path + (key,))
        else:
            base[key] = value


def _apply_override(cfg, assignment):
    if "=" not in assignment:
        raise ValueError(f"Override must look like section.key=value, got: {assignment}")
    dotted, raw = assignment.split("=", 1)
    keys = dotted.split(".")
    nested = yaml.safe_load(raw)
    for key in reversed(keys):
        nested = {key: nested}
    _merge(cfg, nested)


def load_config(path=None, overrides=()):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for assignment in overrides:
        _apply_override(cfg, assignment)
    # YAML 1.1 reads "1e-4" as a string; make sure numeric options are numbers.
    for section, key in FLOAT_KEYS:
        cfg[section][key] = float(cfg[section][key])
    return cfg


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
