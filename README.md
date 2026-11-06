# EchoTrain

Training toolkit for bioacoustic classifiers: CNN and transformer models for
species, call types, dialects and other sound classes.

EchoTrain trains a sound classifier from labelled audio chunks. The labels can be
anything you can hear: species (bats, birds, frogs, insects, ...), but equally
call types, song types, dialects, individuals, life stages, sexes or
behaviours. The model takes raw waveforms, computes a log-mel spectrogram on
the GPU, and predicts which classes are present (multi-label, one sigmoid per
class). Optionally, a second head predicts a coarser parent group of each
class (for species: the genus).

You can train two families of models, both starting from pre-trained weights:

| family | default | other choices |
|---|---|---|
| CNN | **EfficientNet-B3** (ImageNet weights) | any timm EfficientNet (B0 to B8, V2); EfficientNet-B3 with Perch v2 weights |
| Transformer | **PaSST** (AudioSet weights) | 10 s, 20 s or 30 s checkpoint, picked automatically from the chunk length |

The model is one line in the config (`model.name`, see [section 4](#4-models)).
To train your own architecture, write it as a PyTorch module and register it
under a new name; see [Adding your own model](#adding-your-own-model).

> **About the names.** The code grew out of species classification, so some
> option names say `species` and `genus`. Read them as *class* and *parent
> group*: `data.species_list` is the list of classes, and the "genus" head
> predicts whatever grouping `taxonomy.genus_map` defines. See
> [Not only species](#not-only-species) below.

```
train.py                  train a model from a config file
evaluate.py               score a trained model on another split (e.g. test)
configs/example_folders.yaml   example config for a dataset stored as one folder per class
configs/example_manifest.yaml  example config for a dataset described by a manifest CSV
configs/bats.yaml         the European bat setup (manifest dataset, species + genus)
echotrain/                the training code
  options.py              every config option with its default value
  data.py                 reading folders / manifests, class list, genus mapping
  dataset.py              audio loading, length fixing, mixup and noise augmentation
  sampler.py              class-balanced sampling
  frontend.py             waveform -> log-mel spectrogram (Perch v2 style)
  model_zoo.py            model names usable in the config
  networks.py             EfficientNet (any size) and PaSST model classes
  engine.py, metrics.py   training loop, evaluation, mAP / F1 / AUROC
tools/prepare_chunks.py   cut long recordings into fixed-length chunks
tools/add_noise_to_manifest.py  add a background-noise class to a manifest
pyproject.toml, uv.lock   dependencies (managed with uv)
checkpoints/              Perch v2 backbone weights for the perch_* models
legacy/                   unchanged model definitions from the old repository
inputs/bats/              species list and species -> genus table for the bats
```

## 1. Install

Everything runs through [uv](https://docs.astral.sh/uv/), which installs the
right Python version and all packages into a local `.venv/`. Nothing else is
needed: no conda and no manual virtualenv.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once per machine, if uv is not installed yet
git clone https://github.com/bghani/echotrain.git
cd echotrain
uv sync                                            # creates .venv/ with the exact versions in uv.lock
```

After that, put `uv run` in front of every command (`uv run train.py ...`);
it uses the project environment automatically, so there is nothing to
activate. `uv sync` again after pulling changes keeps the environment in
line with `uv.lock`.

- Python: 3.10 to 3.12 (`.python-version` picks 3.11; uv downloads it if needed).
- GPU: on Linux, PyTorch is installed from the CUDA 12.1 wheels, which work with
  any NVIDIA driver >= 525 (`nvidia-smi` shows the driver version). On macOS the
  normal wheels are used (CPU / Apple GPU).
- Adding a package: `uv add <package>`. This updates `pyproject.toml` and `uv.lock`; commit both.
- Linting: `uv run --group dev ruff check .`

The first run downloads the pre-trained weights: ImageNet for EfficientNet (via `timm`), AudioSet for PaSST.
The Perch v2 weights are in `checkpoints/`.

## 2. Prepare the data

The model trains on chunks of a fixed length (`audio.clip_seconds`, 20 s for
bats). A file that is longer is cropped, so the rest of it is never seen; a
file that is shorter is repeated (tiled) to fill the length. The training data
should therefore already be cut into chunks of about `audio.clip_seconds`. If
your recordings are whole (longer files), cut them first, see
[Cutting whole recordings into chunks](#cutting-whole-recordings-into-chunks).

There are two ways to describe a dataset of chunks, set with `data.mode`. Both
train the same way; only the `data:` section of the config differs. Start
from the matching example config:

| `data.mode` | your data | example config |
|---|---|---|
| `folders` | one folder per class (species, call type, ...), separate `train/` and `val/` folders | `configs/example_folders.yaml` |
| `manifest` | one CSV listing every file with its label and split | `configs/example_manifest.yaml` |

In evaluation, the chunks of one recording are max-pooled into a single
prediction, so scores are per recording, not per chunk. How the recording of
a chunk is found is explained under each option.

### Option A: class folders

```
train/Myotis_myotis/rec1_chunk000.wav
train/Myotis_myotis/rec1_chunk001.wav
train/Nyctalus_noctula/rec3_chunk000.wav
val/Myotis_myotis/rec4_chunk000.wav
...
```

Folder names are the class names. Set `data.mode: folders`, `data.train_dir`
and `data.val_dir` (see `configs/example_folders.yaml`). The recording of a
chunk is its file name with the `_chunk003` / `_3` suffix removed
(`eval.group_regex`), taken per class folder: `Myotis_myotis/rec1_chunk000.wav`
and `Myotis_myotis/rec1_chunk001.wav` are pooled as `Myotis_myotis/rec1`.

### Option B: a manifest CSV

One row per chunk with at least three columns: the file path, the label and
the split (`train` / `val` / `test`). Set `data.mode: manifest`,
`data.manifest` and `data.audio_root` (see `configs/example_manifest.yaml`).
Column names are set in the config (`path_col`, `label_col`, `split_col`).
Paths can be absolute or relative to `data.audio_root`. Labels may contain
spaces ("Myotis myotis"); they are turned into `Myotis_myotis`.

Set `data.group_col` to a column that names the source recording of each
chunk (for the bats: `file_path`); chunks with the same value are pooled in
evaluation. Without it, the recording is guessed from the file name as in
Option A.

To add a background-noise class (files in `noise/train`, `noise/val`, `noise/test`):

```bash
uv run tools/add_noise_to_manifest.py --manifest manifest_chunks.csv \
    --noise-dir /data/shared/data/noise --audio-root /data/shared/data --output manifest_with_noise.csv
```

### Cutting whole recordings into chunks

`tools/prepare_chunks.py` cuts every recording into chunks of `--chunk-seconds`
(default 20; use the same value as `audio.clip_seconds`). It reads either
class folders (`--input-dir`) or a manifest of recordings (`--manifest`), and
writes two things:

1. **Chunk files** in `--output-dir`: mono `.wav` at `--target-sr` (default
   32000, the same as `audio.sample_rate`), named `<recording>_chunk000.wav`,
   `<recording>_chunk001.wav`, ..., in one subfolder per class (or all in one
   folder with `--flat`). A leftover at the end shorter than one chunk is tiled to full
   length if it is at least `--min-chunk-seconds` (default 2) long, and dropped
   otherwise. A recording shorter than one chunk always gives one tiled chunk.
2. **A manifest CSV** (`--output-manifest`): one row per chunk, with every
   column of the input plus `chunk_path`, the chunk's path relative to the
   parent folder of `--output-dir`. For folder input the columns are
   `file_path` (the source recording), `latin` (the class), `split` (from
   `--split`) and `chunk_path`.

Which option the output feeds:

| input | use with | how |
|---|---|---|
| class folders (`--input-dir`) | **Option A** | the `--output-dir` is a class-folder tree: use it as `data.train_dir` / `data.val_dir`. The CSV is not needed. |
| a manifest of recordings (`--manifest`) | **Option B** | use the CSV as `data.manifest`, with `data.audio_root` = parent of `--output-dir`, `path_col: chunk_path` and `group_col: file_path`. The chunk folders alone are not enough, because they are not split into train and val. |

**From class folders (Option A).** Run it once per split:

```bash
uv run tools/prepare_chunks.py --input-dir /data/frogs/train --split train \
    --output-dir /data/frogs/chunks/train --output-manifest /data/frogs/chunks/train.csv --chunk-seconds 5
uv run tools/prepare_chunks.py --input-dir /data/frogs/val --split val \
    --output-dir /data/frogs/chunks/val --output-manifest /data/frogs/chunks/val.csv --chunk-seconds 5
```

Then set `data.train_dir: /data/frogs/chunks/train` and
`data.val_dir: /data/frogs/chunks/val` (with `audio.clip_seconds: 5`).

**From a manifest of recordings (Option B).** This is how the bat data was
made. The `tef` column marks real-time ultrasound recordings (`tef == 1`),
which are time-expanded 10x before resampling, the way a bat detector does
it, so that the calls fall into the audible range; rows with `tef == 10` are
already expanded. In folder mode, `--expand-all` expands every file.

```bash
uv run tools/prepare_chunks.py --manifest recordings.csv --audio-root /data/shared \
    --output-dir /data/shared/data/chunks_new --output-manifest manifest_chunks_new.csv \
    --time-expansion-col tef --flat
```

Then set `data.manifest: manifest_chunks_new.csv`,
`data.audio_root: /data/shared/data`, `data.path_col: chunk_path` and
`data.group_col: file_path`. The `split` column of `recordings.csv` is kept,
so every chunk stays in the split of its recording.

### Which classes are trained

By default, every class found in the training data. To fix the set, give
`data.species_list`: a text file with one name per line. Files of other
classes are then ignored, and the run prints how many were skipped. Classes
are always sorted alphabetically; that order is the model's output order and
is saved in `labels.json`.

### Not only species

Nothing in the training depends on the labels being species. A class is just
a folder name (folders mode) or a value of `data.label_col` (manifest mode),
so the same setup trains, for example:

| task | classes (folder names or label values) | parent group for the second head |
|---|---|---|
| species | `Myotis_myotis`, `Nyctalus_noctula` | genus (`Myotis`, `Nyctalus`) |
| call types | `Pipistrellus_pipistrellus_social`, `Pipistrellus_pipistrellus_echolocation` | species |
| song types / dialects | `dialect_north`, `dialect_south` | region, or none |
| life stages | `Rana_temporaria_tadpole`, `Rana_temporaria_adult` | species |
| individuals | `bird_017`, `bird_042` | colony or population, or none |

In manifest mode one CSV can hold several label columns (`latin`,
`call_type`, `dialect`, ...); point `data.label_col` at the one to train on.

Two things to set for non-species labels:

- **Parent group.** By default the second head's group is the class name up
  to the first `_`, which is the genus for `Genus_species` names but usually
  wrong for anything else (`dialect_north` -> `dialect`). Either turn it off
  (`taxonomy.genus_head: false` with a `*_single` model) or give
  `taxonomy.genus_map` with the grouping you want (the CSV header stays
  `species,genus`; the values are your classes and their groups).
- **Recordings with several classes.** In evaluation, the chunks of one
  recording are max-pooled into one prediction, and their labels are merged
  the same way. A recording with chunks of two call types is then scored as
  "contains both". In folders mode, chunks are grouped per class and
  recording, so each class of a recording is scored separately. To score
  every chunk on its own instead (manifest mode), set `data.group_col` to the
  same column as `data.path_col`.

### Parent-group (genus) head (optional)

With `taxonomy.genus_head: true`, the model also learns the genus, or any
other parent group of the classes. By default
the genus is the first word of the class name (`Myotis_myotis` -> `Myotis`),
and a one-word class such as `noise` is its own genus. For anything else,
give `taxonomy.genus_map`: a CSV with columns `species,genus`. The "genus" can
be any higher rank, e.g. family.

## 3. Train

Copy the example config for your data, change the paths, and train.

**Folders mode** (`data.mode: folders`):

```bash
cp configs/example_folders.yaml configs/my_folders.yaml    # then set data.train_dir / data.val_dir
uv run train.py --config configs/my_folders.yaml
uv run evaluate.py --run runs/my_species_group --dir /path/to/test   # test set = another species-folder tree
```

**Manifest mode** (`data.mode: manifest`):

```bash
cp configs/example_manifest.yaml configs/my_manifest.yaml  # then set data.manifest / data.audio_root / column names
uv run train.py --config configs/my_manifest.yaml
uv run evaluate.py --run runs/my_species_group --split test           # test set = the rows with split "test"
```

`my_species_group` is the `run_name` in the example configs; use your own.
Complete real setups: `configs/bats.yaml` (manifest mode) and
`configs/frogs.yaml` (folders mode).

Change any option from the command line without editing the file:

```bash
uv run train.py --config configs/bats.yaml --set run_name=bats_lr3e-4 --set train.lr=3e-4 --set device=cuda:1
```

All options and their defaults are in `echotrain/options.py`, with a comment
for each. A config file only needs the values that differ from the defaults.

Each run writes to `runs/<run_name>/`. If that folder already holds a run, a new
one is started next to it as `runs/<run_name>_<model>_<timestamp>/` (e.g.
`frogs_passt_20261106-143012`); an existing run is never overwritten. Pass
`--set run_name=...` for a readable name, or `--resume` to continue the old run.

| file | contents |
|---|---|
| `config.yaml` | the complete config of the run |
| `labels.json` | class (and genus) names in model output order |
| `best_model.pt` | weights of the epoch with the highest validation mAP |
| `last.pt` | latest weights and optimizer state |
| `history.csv` | metrics for every epoch |
| `best_val/` | for the best epoch: AP per class, most frequent confusions, per-recording scores |

If a run stops (crash, time limit), continue it with:

```bash
uv run train.py --resume runs/bats_effnet_b3/last.pt
```

**Model selection.** The best epoch is picked on the `val` split. Keep the
`test` split for the end:

```bash
uv run evaluate.py --run runs/bats_effnet_b3 --split test
uv run evaluate.py --run runs/my_frogs --dir /data/frogs/test        # folder dataset
```

**Starting from an earlier model.** `model.init_checkpoint: path/to/best_model.pt`
loads every weight whose name and shape match. With a different class list,
the backbone is reused and the output heads start fresh.

## 4. Models

Set `model.name` to one of the names below (default `effnet`). In this table
"species" means the class head and "genus" the parent-group head.

| name | heads | description |
|---|---|---|
| `effnet` | species + genus | EfficientNet. Default. |
| `effnet_single` | species | EfficientNet, no genus head |
| `passt` | species + genus | PaSST audio transformer (AudioSet weights) |
| `passt_single` | species | PaSST, no genus head |
| `perch_effnet_b3` | species + genus | EfficientNet-B3 starting from Perch v2 weights (`model.perch_weights`) |
| `perch_effnet_b3_single` | species | Perch v2 weights, no genus head |

Models with a genus head need `taxonomy.genus_head: true`; the `*_single`
models need it to be `false`.

**Perch weights.** The `perch_*` models load their backbone from
`model.perch_weights`, by default `checkpoints/perch_v2_backbone_timm.pt`,
which ships with the repository. Set it to `null` to start from random weights.

**EfficientNet size.** The `effnet*` models take their backbone from
`model.backbone` (default `efficientnet_b3`). Any EfficientNet in `timm`
works, e.g. `efficientnet_b0` to `efficientnet_b5`, `tf_efficientnet_b0` to
`tf_efficientnet_b8`, or `tf_efficientnetv2_s`. With `model.pretrained: true`
(the default) the backbone must have ImageNet weights in timm: for B6 to B8,
use the `tf_` versions. The old names `effnet_b3` and `effnet_b3_single`
still work and mean the same as `effnet` / `effnet_single` with
`efficientnet_b3`.

```bash
uv run train.py --config configs/bats.yaml --set model.backbone=efficientnet_b0 --set run_name=bats_b0
```

**PaSST.** `model.passt_arch` picks the pre-trained AudioSet weights. PaSST
checkpoints are trained on a fixed input length:

| `model.passt_arch` | pre-training length | chosen by `auto` when `audio.clip_seconds` is |
|---|---|---|
| `passt_s_kd_p16_128_ap486` | 10 s | 15 or less |
| `passt_20sec` | 20 s | more than 15, up to 25 |
| `passt_30sec` | 30 s | more than 25 |

By default (`passt_arch: auto`) the checkpoint whose length is closest to
`audio.clip_seconds` is used, and the run prints which one, e.g.
`model.passt_arch=auto -> passt_s_kd_p16_128_ap486 (closest to clip_seconds=5)`.
Other chunk lengths work too: the time position encodings are cut or
stretched to fit. To use a specific checkpoint, name it in the config or on
the command line:

```bash
uv run train.py --config configs/frogs.yaml --set model.name=passt --set model.passt_arch=passt_20sec
```

Besides the three above, `PASST_ARCHS` in `echotrain/networks.py` lists a few
more 10 s checkpoints (`passt_s_swa_p16_128_ap476`, `passt_l_kd_p16_128_ap47`,
`openmic`, `fsd50k`); `auto` never picks these. PaSST uses its own spectrogram (the one
it was pre-trained with: 128 mel bins, 10 ms hop, from `model.passt_fmin` to
`model.passt_fmax`). Only `frontend.freqm` / `frontend.timem` (masking) apply
from the `frontend:` section. The weights (about 350 MB) are downloaded on the
first run. A PaSST checkpoint from the old batecho / insectaecho code can be
used as `model.init_checkpoint`.

PaSST needs much more GPU memory than EfficientNet. On a 48 GB GPU with 20 s
chunks, use `train.batch_size: 2` with `train.grad_accum_steps: 4` or more.
You can also turn on structured patchout (`model.passt_patchout_t: 100`,
`model.passt_patchout_f: 4`), which drops patches in training and allows a
batch size of 8.

```bash
uv run train.py --config configs/bats.yaml --set model.name=passt --set run_name=bats_passt \
    --set train.batch_size=2 --set train.grad_accum_steps=8
```

**Spectrograms.** The EfficientNet and Perch models use the shared frontend
(`echotrain/frontend.py`, settings under `frontend:`); PaSST uses its own
(see above). With a 32 kHz sample rate the shared frontend covers 60 Hz to
16 kHz. Bat recordings are time-expanded 10x during data preparation, so this
range corresponds to 0.6 to 160 kHz in the original recording.

### Adding your own model

Any PyTorch model can be trained, CNN or transformer. It has to:

- take a batch of raw waveforms, shape `(batch, samples)`, at
  `audio.sample_rate`, and compute its own spectrogram (reuse
  `echotrain/frontend.py`, as the EfficientNet models do, or bring your own);
- return the class logits `(batch, n_classes)`, or a tuple
  `(class_logits, parent_logits)` if it has a parent-group (genus) head.

Then:

1. Put the class in `echotrain/networks.py` (next to the EfficientNet and
   PaSST classes, which are good templates).
2. Add a builder function and an entry to `MODELS` in
   `echotrain/model_zoo.py`:

   ```python
   def _my_net(m, cfg, n_species, n_genus):
       return networks.MyNet(n_species, n_genus, mel=_frontend(cfg))

   MODELS = {
       ...
       # name: (builder, two_heads, needs_legacy, description)
       "my_net": (_my_net, True, False, "My architecture. Species + genus heads."),
   }
   ```

   `two_heads` says whether the model returns the parent-group logits too
   (use it with `taxonomy.genus_head: true`). New settings for your model go
   into the `model:` section of `DEFAULTS` in `echotrain/options.py`.
3. Train it with `--set model.name=my_net`. Loss, sampling, mixup,
   evaluation and checkpoints work unchanged.

`legacy/models.py` is an unchanged copy from the old repository with many
more architectures (it is where the Perch models come from); they can be
registered the same way with `needs_legacy=True`.

## 5. Training details

- Loss: binary cross-entropy per class; with a genus head, plus
  `train.genus_loss_weight` times the genus BCE.
- Optimizer: AdamW (`train.lr` 1e-4, weight decay 1e-4), mixed precision,
  gradient clipping at 1.0. The learning rate drops 10x when the validation
  mAP has not improved for `train.plateau_patience` epochs.
- Class balance: with `sampler.balanced`, files of rare classes are drawn
  more often (weight 1 / min(count, cap)^power). `sampler.noise_class` is
  drawn less often.
- Augmentation: mixup of two training chunks (`augment.mixup_rate`; the mix
  is labelled with both classes), SpecAugment masking in the frontend, and
  optionally coloured noise, time shift and background noise.
- Metrics, after pooling chunks into recordings: mAP (mean over classes that
  have at least one validation recording), weighted F1 at
  `eval.threshold`, weighted AUROC.

Out of GPU memory? Lower `train.batch_size` and raise
`train.grad_accum_steps` by the same factor.
