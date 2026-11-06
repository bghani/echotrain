"""Collect audio files and labels from species folders or from a manifest CSV.

Both sources produce the same thing: a `Split` holding, per audio file, its
path, its class name and a group key that names the recording it was cut
from (used to pool chunk predictions during evaluation).
"""
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Split:
    paths: list
    labels: list   # class names, e.g. "Myotis_myotis"
    groups: list   # recording each file belongs to
    species_idx: list = field(default_factory=list)
    genus_idx: list = field(default_factory=list)

    def __len__(self):
        return len(self.paths)


def normalize_label(name):
    """'Myotis myotis ' -> 'Myotis_myotis'. Folder names and manifest labels end up identical."""
    return re.sub(r"\s+", "_", str(name).strip())


def _group_from_path(path, group_regex):
    return re.sub(group_regex, "", Path(path).stem)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_folders(root, extensions, group_regex):
    """root/<species>/<file>. Each subfolder name is a class."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")
    extensions = {e.lower() for e in extensions}
    paths, labels, groups = [], [], []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = normalize_label(class_dir.name)
        for f in sorted(class_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in extensions:
                paths.append(str(f))
                labels.append(label)
                groups.append(f"{label}/{_group_from_path(f, group_regex)}")
    if not paths:
        raise RuntimeError(f"No audio files with extensions {sorted(extensions)} under {root}")
    return Split(paths, labels, groups)


def read_manifest(data_cfg, split_values, group_regex):
    """Rows of the manifest whose split column is one of `split_values`."""
    path_col, label_col, split_col = data_cfg["path_col"], data_cfg["label_col"], data_cfg["split_col"]
    group_col = data_cfg["group_col"]
    df = pd.read_csv(data_cfg["manifest"], low_memory=False)
    needed = [path_col, label_col, split_col] + ([group_col] if group_col else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Manifest {data_cfg['manifest']} has no column(s) {missing}")

    df = df[df[split_col].astype(str).isin([str(s) for s in split_values])]
    df = df.dropna(subset=[path_col, label_col])
    if df.empty:
        raise RuntimeError(f"No manifest rows with {split_col} in {list(split_values)}")

    root = data_cfg["audio_root"]
    paths = [p if (root is None or os.path.isabs(p)) else os.path.join(root, p)
             for p in df[path_col].astype(str)]
    labels = [normalize_label(x) for x in df[label_col]]
    if group_col:
        groups = [str(g) if pd.notna(g) else _group_from_path(p, group_regex)
                  for g, p in zip(df[group_col], paths, strict=True)]
    else:
        groups = [_group_from_path(p, group_regex) for p in paths]
    return Split(paths, labels, groups)


def read_split(cfg, which):
    """which = "train" or "val" (uses the config), or a list of manifest split values."""
    data_cfg, group_regex = cfg["data"], cfg["eval"]["group_regex"]
    if data_cfg["mode"] == "folders":
        root = data_cfg[f"{which}_dir"] if which in ("train", "val") else which
        return read_folders(root, data_cfg["audio_extensions"], group_regex)
    if data_cfg["mode"] == "manifest":
        values = data_cfg[f"{which}_splits"] if which in ("train", "val") else which
        return read_manifest(data_cfg, values, group_regex)
    raise ValueError(f"data.mode must be 'folders' or 'manifest', got {data_cfg['mode']!r}")


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

@dataclass
class LabelSpace:
    species: list                 # sorted class names; index = model output index
    genus: list = None            # sorted genus names, or None without a genus head
    species_to_genus: dict = None

    @property
    def has_genus(self):
        return self.genus is not None

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"species": self.species, "genus": self.genus,
                       "species_to_genus": self.species_to_genus}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["species"], d.get("genus"), d.get("species_to_genus"))


def read_species_list(path):
    with open(path) as f:
        names = [normalize_label(line) for line in f if line.strip()]
    dupes = [n for n, c in Counter(names).items() if c > 1]
    if dupes:
        raise ValueError(f"Duplicate entries in {path}: {dupes}")
    return names


def build_label_space(cfg, train_split):
    if cfg["data"]["species_list"]:
        species = sorted(read_species_list(cfg["data"]["species_list"]))
    else:
        species = sorted(set(train_split.labels))

    if not cfg["taxonomy"]["genus_head"]:
        return LabelSpace(species)

    genus_map = {}
    if cfg["taxonomy"]["genus_map"]:
        df = pd.read_csv(cfg["taxonomy"]["genus_map"])
        genus_map = {normalize_label(s): normalize_label(g) for s, g in zip(df["species"], df["genus"], strict=True)}
    # Without an entry in the map, the genus is the first word of the name
    # ("Myotis_myotis" -> "Myotis"); a one-word class such as "noise" is its own genus.
    species_to_genus = {s: genus_map.get(s, s.split("_")[0]) for s in species}
    genus = sorted(set(species_to_genus.values()))
    return LabelSpace(species, genus, species_to_genus)


def encode(split, label_space, name="split"):
    """Drop files whose class is not in the label space and attach class indices."""
    sp_index = {s: i for i, s in enumerate(label_space.species)}
    keep = [i for i, lab in enumerate(split.labels) if lab in sp_index]
    dropped = Counter(lab for lab in split.labels if lab not in sp_index)
    if dropped:
        print(f"[{name}] ignoring {sum(dropped.values())} files from {len(dropped)} classes "
              f"not in the class list: {dict(dropped.most_common(10))}")
    out = Split([split.paths[i] for i in keep], [split.labels[i] for i in keep],
                [split.groups[i] for i in keep])
    out.species_idx = [sp_index[lab] for lab in out.labels]
    if label_space.has_genus:
        g_index = {g: i for i, g in enumerate(label_space.genus)}
        out.genus_idx = [g_index[label_space.species_to_genus[lab]] for lab in out.labels]
    if not out.paths:
        raise RuntimeError(f"[{name}] no files left after matching the class list")
    return out


def describe(split, name):
    counts = Counter(split.labels)
    print(f"[{name}] {len(split)} files, {len(set(split.groups))} recordings, {len(counts)} classes "
          f"(smallest: {min(counts.values())}, largest: {max(counts.values())})")
