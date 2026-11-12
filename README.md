# EchoTrain

Train sound classifiers from labelled audio chunks: species (bats, birds,
frogs, insects, ...), call types, dialects, individuals, life stages or any
other class you can hear. The model takes raw waveforms, computes a log-mel
spectrogram on the GPU and predicts which classes are present (multi-label).
An optional second head predicts a coarser parent group of each class (for
species: the genus).

Two model families, both starting from pre-trained weights:

| family | default | other choices |
|---|---|---|
| CNN | **EfficientNet-B3** (ImageNet weights) | any timm EfficientNet (B0 to B8, V2); EfficientNet-B3 with Perch v2 weights |
| Transformer | **PaSST** (AudioSet weights) | 10 s, 20 s or 30 s checkpoint, picked automatically from the chunk length |

Advanced options, internals and how to add your own model are in
[DEVELOPER.md](DEVELOPER.md).

## 1. Install

Everything runs through [uv](https://docs.astral.sh/uv/), which installs the
right Python version and all packages into a local `.venv/`. No conda, no
manual virtualenv.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once per machine, if uv is not installed yet
git clone https://github.com/bghani/echotrain.git
cd echotrain
uv sync                                            # creates .venv/ with the exact versions in uv.lock
```

Put `uv run` in front of every command (`uv run train.py ...`); there is
nothing to activate. Run `uv sync` again after pulling changes.

On Windows, install uv with
`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.

**GPU.** Training uses the first GPU it finds, else the CPU (`device` in the config):

| Machine | GPU used | Notes |
|---|---|---|
| Linux or Windows with an NVIDIA GPU | CUDA | `uv sync` installs the CUDA 12.1 build of PyTorch; needs an NVIDIA driver >= 525 |
| Mac with Apple Silicon (M1–M4) | MPS | no mixed precision; if an operation is missing on MPS, set `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| anything else | CPU | works, but slowly |

### Quick start: the tutorial

`notebooks/tutorial_toy_frogs.ipynb` runs the whole workflow on a small frog
dataset that ships with the repository (`data/toy_frogs`: 70 field
recordings of 5 species): exploring the sounds, cutting the recordings into
chunks, training and comparing two models, evaluating on a test set, and UMAP
maps of the data before and after training. It takes a few minutes on a GPU.

```bash
uv sync --group notebook
uv run --group notebook jupyter lab notebooks/tutorial_toy_frogs.ipynb
```

The same without the notebook:

```bash
for s in train val test; do
  uv run tools/prepare_chunks.py --input-dir data/toy_frogs/recordings/$s --split $s \
    --output-dir data/toy_frogs_chunks/$s --output-manifest data/toy_frogs_chunks/$s.csv --chunk-seconds 5
done
uv run train.py --config configs/toy_frogs.yaml
uv run evaluate.py --run runs/toy_frogs --dir data/toy_frogs_chunks/test
```

## 2. Prepare the data

The model trains on chunks of a fixed length (`audio.clip_seconds`). Longer
files are cropped and shorter ones repeated, so the training data should
already be cut into chunks of about that length. If your recordings are
whole, cut them first (see [Cutting recordings into chunks](#cutting-recordings-into-chunks)).

There are two ways to describe a dataset, set with `data.mode`. Start from the
matching example config:

| `data.mode` | your data | example config |
|---|---|---|
| `folders` | one folder per class, separate `train/` and `val/` folders | `configs/example_folders.yaml` |
| `manifest` | one CSV listing every file with its label and split | `configs/example_manifest.yaml` |

In evaluation, the chunks of one recording are max-pooled into a single
prediction, so scores are per recording, not per chunk.

### Option A: class folders

```
train/Myotis_myotis/rec1_chunk000.wav
train/Myotis_myotis/rec1_chunk001.wav
train/Nyctalus_noctula/rec3_chunk000.wav
val/Myotis_myotis/rec4_chunk000.wav
...
```

Folder names are the class names. Set `data.mode: folders`, `data.train_dir`
and `data.val_dir`. The recording of a chunk is its file name without the
`_chunk003` / `_3` suffix.

### Option B: a manifest CSV

One row per chunk with at least the file path, the label and the split
(`train` / `val` / `test`). Set `data.mode: manifest`, `data.manifest` and
`data.audio_root`; column names are set with `path_col`, `label_col`,
`split_col`. Paths can be absolute or relative to `data.audio_root`. Set
`data.group_col` to a column naming the source recording of each chunk, so
its chunks are pooled in evaluation.

### Cutting recordings into chunks

`tools/prepare_chunks.py` cuts every recording into chunks of
`--chunk-seconds` (use the same value as `audio.clip_seconds`) and writes
the chunk files plus a manifest CSV.

From class folders (gives an Option A tree), once per split:

```bash
uv run tools/prepare_chunks.py --input-dir /data/frogs/train --split train \
    --output-dir /data/frogs/chunks/train --output-manifest /data/frogs/chunks/train.csv --chunk-seconds 5
uv run tools/prepare_chunks.py --input-dir /data/frogs/val --split val \
    --output-dir /data/frogs/chunks/val --output-manifest /data/frogs/chunks/val.csv --chunk-seconds 5
```

Then set `data.train_dir: /data/frogs/chunks/train`,
`data.val_dir: /data/frogs/chunks/val` and `audio.clip_seconds: 5`.

From a manifest of recordings (gives an Option B manifest):

```bash
uv run tools/prepare_chunks.py --manifest recordings.csv --audio-root /data/shared \
    --output-dir /data/shared/data/chunks --output-manifest manifest_chunks.csv --flat
```

Then use the new CSV as `data.manifest`, with `data.audio_root` = parent of
`--output-dir`, `data.path_col: chunk_path` and `data.group_col: file_path`.
Every chunk keeps the split of its recording.

More options (output layout, leftover handling, time expansion for bat
recordings, adding a noise class) are in [DEVELOPER.md](DEVELOPER.md#data-preparation-in-detail).

### Which classes are trained

By default, every class found in the training data. To fix the set, give
`data.species_list`: a text file with one name per line. Classes are sorted
alphabetically; that is the model's output order, saved in `labels.json`.

### Not only species

The code grew out of species classification, so some options say `species`
and `genus`. Read them as *class* and *parent group*: `data.species_list` is
the list of classes, and the "genus" head predicts whatever grouping
`taxonomy.genus_map` defines. For non-species labels (call types, dialects,
...), either turn the parent head off (`taxonomy.genus_head: false` with a
`*_single` model) or give a `taxonomy.genus_map`, because the default group
(the class name up to the first `_`) only makes sense for `Genus_species`
names. Examples and details: [DEVELOPER.md](DEVELOPER.md#not-only-species).

## 3. Train and evaluate

Copy the example config for your data, set the paths, train, then evaluate on
the test set:

```bash
# folders mode
cp configs/example_folders.yaml configs/my_folders.yaml    # set data.train_dir / data.val_dir
uv run train.py --config configs/my_folders.yaml
uv run evaluate.py --run runs/my_species_group --dir /path/to/test   # test set = another class-folder tree

# manifest mode
cp configs/example_manifest.yaml configs/my_manifest.yaml  # set data.manifest / data.audio_root / column names
uv run train.py --config configs/my_manifest.yaml
uv run evaluate.py --run runs/my_species_group --split test          # test set = rows with split "test"
```

`my_species_group` is the `run_name` in the example configs; use your own.
Complete real setups: `configs/bats.yaml` (manifest) and `configs/frogs.yaml`
(folders).

Override any option from the command line:

```bash
uv run train.py --config configs/bats.yaml --set run_name=bats_lr3e-4 --set train.lr=3e-4 --set device=cuda:1
```

All options and their defaults are in `echotrain/options.py`, with a comment
for each; a config only needs the values that differ.

Each run writes to `runs/<run_name>/` (an existing run is never overwritten;
a new folder with a timestamp is made next to it):

| file | contents |
|---|---|
| `config.yaml` | the complete config of the run |
| `labels.json` | class (and genus) names in model output order |
| `best_model.pt` | weights of the epoch with the highest validation mAP |
| `last.pt` | latest weights and optimizer state |
| `history.csv` | metrics for every epoch |
| `best_val/` | for the best epoch: AP per class, most frequent confusions, per-recording scores |

Continue a stopped run with `uv run train.py --resume runs/<run>/last.pt`.
The best epoch is picked on `val`; keep `test` for the end.

Out of GPU memory? Lower `train.batch_size` and raise
`train.grad_accum_steps` by the same factor.

## 4. Models

Set `model.name` (default `effnet`). "species" means the class head and
"genus" the parent-group head.

| name | heads | description |
|---|---|---|
| `effnet` | species + genus | EfficientNet. Default. |
| `effnet_single` | species | EfficientNet, no genus head |
| `passt` | species + genus | PaSST audio transformer (AudioSet weights) |
| `passt_single` | species | PaSST, no genus head |
| `perch_effnet_b3` | species + genus | EfficientNet-B3 starting from Perch v2 weights |
| `perch_effnet_b3_single` | species | Perch v2 weights, no genus head |

Models with a genus head need `taxonomy.genus_head: true`; the `*_single`
models need it `false`.

- **EfficientNet size:** `model.backbone` (default `efficientnet_b3`), e.g.
  `--set model.backbone=efficientnet_b0`.
- **PaSST checkpoint:** `model.passt_arch: auto` (default) picks the 10 s,
  20 s or 30 s checkpoint closest to `audio.clip_seconds`. PaSST needs much
  more GPU memory: with 20 s chunks on a 48 GB GPU use
  `train.batch_size: 2` and `train.grad_accum_steps: 4` or more.
- **Start from an earlier model:** `model.init_checkpoint: path/to/best_model.pt`.

The first run downloads the pre-trained weights (ImageNet via `timm`, AudioSet
for PaSST); the Perch v2 weights are in `checkpoints/`. Model details and how
to add your own architecture: [DEVELOPER.md](DEVELOPER.md#models-in-detail).
