# echo_train

Train a sound classifier for any species group (bats, birds, frogs, insects, ...)
from labelled audio clips. The model takes raw waveforms, computes a log-mel
spectrogram on the GPU, and predicts which species are present (multi-label,
one sigmoid per class). Optionally, a second head predicts the genus.

```
train.py                  train a model from a config file
evaluate.py               score a trained model on another split (e.g. test)
configs/bats.yaml         the European bat setup (manifest dataset, species + genus)
configs/folders_example.yaml  template for a dataset stored as species folders
echotrain/                the training code
  options.py              every config option with its default value
  data.py                 reading folders / manifests, class list, genus mapping
  dataset.py              audio loading, length fixing, mixup and noise augmentation
  sampler.py              class-balanced sampling
  frontend.py             waveform -> log-mel spectrogram (Perch v2 style)
  model_zoo.py            model names usable in the config
  engine.py, metrics.py   training loop, evaluation, mAP / F1 / AUROC
tools/prepare_chunks.py   cut long recordings into fixed-length chunks
tools/add_noise_to_manifest.py  add a background-noise class to a manifest
pyproject.toml, uv.lock   dependencies (managed with uv)
legacy/                   unchanged model definitions from the old repository
inputs/bats/              species list and species -> genus table for the bats
```

## 1. Install

Everything runs through [uv](https://docs.astral.sh/uv/), which installs the
right Python version and all packages into a local `.venv/`. Nothing else is
needed: no conda and no manual virtualenv.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once per machine, if uv is not installed yet
cd echo_train
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

The first run downloads ImageNet weights for EfficientNet-B3 (via `timm`).

## 2. Prepare the data

The model trains on clips of a fixed length (`audio.clip_seconds`, 20 s for
bats). Clips that are longer are cropped and shorter ones are repeated
(tiled) to fill the length. Long recordings should first be cut into chunks
so that nothing is thrown away. In evaluation, the chunks of one recording are
max-pooled into a single prediction.

There are two ways to describe a dataset. Both train the same way.

### Option A: species folders

```
train/Myotis_myotis/rec1.wav
train/Myotis_myotis/rec2.wav
train/Nyctalus_noctula/rec3.wav
val/Myotis_myotis/rec4.wav
...
```

Folder names are the class names. Set `data.mode: folders`, `data.train_dir`
and `data.val_dir` (see `configs/folders_example.yaml`). To cut the files into
chunks first:

```bash
uv run tools/prepare_chunks.py --input-dir /data/frogs/train --split train \
    --output-dir /data/frogs/chunks/train --output-manifest /data/frogs/chunks/train.csv --chunk-seconds 5
```

The chunks keep the species folder structure, so `/data/frogs/chunks/train`
can be used directly as `data.train_dir`.

### Option B: a manifest CSV

One row per audio file with at least three columns: the file path, the label
and the split (`train` / `val` / `test`). Column names are set in the config
(`path_col`, `label_col`, `split_col`). Paths can be absolute or relative to
`data.audio_root`. Labels may contain spaces ("Myotis myotis"); they are
turned into `Myotis_myotis`.

If the files are chunks, also set `data.group_col` to a column that names the
source recording (for the bats: `file_path`). Chunks of one recording are then
pooled in evaluation. Without it, the recording is guessed from the file name
by removing a `_chunk003` / `_3` suffix (`eval.group_regex`).

To make chunks from a manifest of full recordings (this is how the bat data
was made; the `tef` column marks real-time ultrasound recordings that need
10x time expansion):

```bash
uv run tools/prepare_chunks.py --manifest recordings.csv --audio-root /data/shared \
    --output-dir /data/shared/data/chunks_new --output-manifest manifest_chunks_new.csv \
    --time-expansion-col tef --flat
```

To add a background-noise class (files in `noise/train`, `noise/val`, `noise/test`):

```bash
uv run tools/add_noise_to_manifest.py --manifest manifest_chunks_new.csv \
    --noise-dir /data/shared/data/noise --audio-root /data/shared/data --output manifest_with_noise.csv
```

### Which classes are trained

By default, every class found in the training data. To fix the set, give
`data.species_list`: a text file with one name per line. Files of other
classes are then ignored, and the run prints how many were skipped. Classes
are always sorted alphabetically; that order is the model's output order and
is saved in `labels.json`.

### Genus head (optional)

With `taxonomy.genus_head: true`, the model also learns the genus. By default
the genus is the first word of the class name (`Myotis_myotis` -> `Myotis`),
and a one-word class such as `noise` is its own genus. For anything else,
give `taxonomy.genus_map`: a CSV with columns `species,genus`. The "genus" can
be any higher rank, e.g. family.

## 3. Train

```bash
uv run train.py --config configs/bats.yaml
```

Change any option from the command line without editing the file:

```bash
uv run train.py --config configs/bats.yaml --set run_name=bats_lr3e-4 --set train.lr=3e-4 --set device=cuda:1
```

All options and their defaults are in `echotrain/options.py`, with a comment
for each. A config file only needs the values that differ from the defaults.

Each run writes to `runs/<run_name>/`:

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
uv run train.py --resume runs/bats_effnet_b3_chunked/last.pt
```

**Model selection.** The best epoch is picked on the `val` split. Keep the
`test` split for the end:

```bash
uv run evaluate.py --run runs/bats_effnet_b3_chunked --split test
uv run evaluate.py --run runs/my_frogs --dir /data/frogs/test        # folder dataset
```

**Starting from an earlier model.** `model.init_checkpoint: path/to/best_model.pt`
loads every weight whose name and shape match. With a different class list,
the backbone is reused and the output heads start fresh.

## 4. Models

Set `model.name` to one of:

| name | heads | description |
|---|---|---|
| `effnet_b3_chunked` | species + genus | EfficientNet-B3 (ImageNet) applied to each `model.chunk_seconds` piece of the clip; the chunk scores are max-pooled. Default for bats. |
| `effnet_b3` | species + genus | EfficientNet-B3 (ImageNet) on the whole clip |
| `perch_effnet_b3` | species + genus | EfficientNet-B3 starting from Perch v2 weights (`model.perch_weights`) |
| `effnet_b3_single` | species | EfficientNet-B3 (ImageNet), no genus head |
| `perch_effnet_b3_single` | species | Perch v2 weights, no genus head |

Models with a genus head need `taxonomy.genus_head: true`; the `*_single`
models need it to be `false`.

All models use the same spectrogram frontend (`echotrain/frontend.py`,
settings under `frontend:`). With a 32 kHz sample rate it covers 60 Hz to
16 kHz. Bat recordings are time-expanded 10x during data preparation, so this
range corresponds to 0.6 to 160 kHz in the original recording.

The model classes live in `legacy/models.py`, an unchanged copy from the old
repository that contains many more architectures. To use another one, add it
to `MODELS` in `echotrain/model_zoo.py`.

## 5. Training details

- Loss: binary cross-entropy per class; with a genus head, plus
  `train.genus_loss_weight` times the genus BCE.
- Optimizer: AdamW (`train.lr` 1e-4, weight decay 1e-4), mixed precision,
  gradient clipping at 1.0. The learning rate drops 10x when the validation
  mAP has not improved for `train.plateau_patience` epochs.
- Class balance: with `sampler.balanced`, files of rare classes are drawn
  more often (weight 1 / min(count, cap)^power). `sampler.noise_class` is
  drawn less often.
- Augmentation: mixup of two training clips (`augment.mixup_rate`; the mix
  is labelled with both classes), SpecAugment masking in the frontend, and
  optionally coloured noise, time shift and background noise.
- Metrics, after pooling chunks into recordings: mAP (mean over classes that
  have at least one validation recording), weighted F1 at
  `eval.threshold`, weighted AUROC.

Out of GPU memory? Lower `train.batch_size` and raise
`train.grad_accum_steps` by the same factor.
