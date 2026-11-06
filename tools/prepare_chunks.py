"""Cut recordings into fixed-length chunks and write a chunk manifest.

Each recording is resampled to --target-sr and split into chunks of
--chunk-seconds. A trailing piece of at least --min-chunk-seconds is tiled up
to full length; shorter tails are dropped. A recording shorter than one chunk
is always tiled, so every recording yields at least one chunk.

Input is either a manifest CSV (one row per recording) or a folder with one
subfolder per species. The output manifest has one row per chunk, keeps every
input column, and adds `chunk_path` (relative to the parent of --output-dir).
Train with data.audio_root set to that parent folder.

Examples
--------
# bats: manifest with split column; tef == 1 rows are real-time ultrasound and get time-expanded 10x
uv run tools/prepare_chunks.py --manifest recordings.csv --audio-root /data/shared \\
    --output-dir /data/shared/data/chunks_v5 --output-manifest inputs/bats/manifest_chunks_v5.csv \\
    --time-expansion-col tef

# any group: species folders, one split at a time
uv run tools/prepare_chunks.py --input-dir /data/frogs/train --split train \\
    --output-dir /data/frogs/chunks --output-manifest /data/frogs/manifest.csv --chunk-seconds 5
"""
import argparse
import logging
import os
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg"}
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prepare_chunks")


def tile_to_length(x, n):
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


def chunk_audio(x, n, n_min):
    n_full = len(x) // n
    chunks = [x[i * n:(i + 1) * n] for i in range(n_full)]
    tail = x[n_full * n:]
    if len(tail) and (n_full == 0 or len(tail) >= n_min):
        chunks.append(tile_to_length(tail, n))
    return chunks


def load_resampled(path, target_sr, expand):
    """Load at native rate and resample to target_sr.

    With expand > 1 (time expansion), the recording is treated as if its rate
    were native_sr / expand, i.e. slowed down so ultrasound becomes audible
    before resampling. This is what bat detectors do in time-expansion mode.
    """
    x, native_sr = librosa.load(path, sr=None, mono=True)
    return librosa.resample(x, orig_sr=native_sr / expand, target_sr=target_sr)


def resolve(path, root):
    p = Path(path) if root is None or os.path.isabs(path) else Path(root) / path
    if not p.exists():  # filenames from macOS volumes are often stored NFD-normalised
        nfd = Path(unicodedata.normalize("NFD", str(p)))
        if nfd.exists():
            return nfd
    return p


def process(row, args):
    src = row["_src"]
    expand = row["_expand"]
    try:
        x = load_resampled(str(src), args.target_sr, expand)
    except Exception as e:
        log.error(f"Failed to load {src}: {e}")
        return []
    n = int(args.chunk_seconds * args.target_sr)
    chunks = chunk_audio(x, n, int(args.min_chunk_seconds * args.target_sr))
    out_dir = Path(args.output_dir) / row["_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, c in enumerate(chunks):
        out = out_dir / f"{src.stem}_chunk{i:03d}.wav"
        sf.write(out, c, args.target_sr)
        rec = {k: v for k, v in row.items() if not k.startswith("_")}
        rec["chunk_path"] = str(out.relative_to(Path(args.output_dir).parent))
        records.append(rec)
    return records


def rows_from_manifest(args):
    df = pd.read_csv(args.manifest, low_memory=False)
    for col in [args.path_col, args.label_col]:
        if col not in df.columns:
            raise SystemExit(f"Manifest has no column {col!r}")
    rows = []
    for row in df.to_dict("records"):
        expand = 1.0
        if args.time_expansion_col:
            tef = float(row[args.time_expansion_col])
            if tef == 1.0:
                expand = args.time_expansion_factor
            elif tef != args.time_expansion_factor:
                log.warning(f"Skipping {row[args.path_col]}: unexpected {args.time_expansion_col}={tef}")
                continue
        row["_src"] = resolve(str(row[args.path_col]), args.audio_root)
        row["_expand"] = expand
        row["_subdir"] = "." if args.flat else str(row[args.label_col]).replace(" ", "_")
        rows.append(row)
    return rows


def rows_from_folders(args):
    rows = []
    for class_dir in sorted(p for p in Path(args.input_dir).iterdir() if p.is_dir()):
        for f in sorted(class_dir.rglob("*")):
            if f.suffix.lower() in AUDIO_EXT:
                rows.append({"file_path": str(f), "latin": class_dir.name, "split": args.split,
                             "_src": f, "_expand": args.time_expansion_factor if args.expand_all else 1.0,
                             "_subdir": "." if args.flat else class_dir.name})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="CSV with one row per recording")
    src.add_argument("--input-dir", help="folder with one subfolder per species")
    p.add_argument("--audio-root", help="manifest paths are relative to this")
    p.add_argument("--path-col", default="file_path")
    p.add_argument("--label-col", default="latin")
    p.add_argument("--split", default="train", help="--input-dir only: value for the split column")
    p.add_argument("--output-dir", required=True, help="where chunk .wav files go")
    p.add_argument("--output-manifest", required=True)
    p.add_argument("--flat", action="store_true", help="all chunks in one folder instead of one per species")
    p.add_argument("--target-sr", type=int, default=32000)
    p.add_argument("--chunk-seconds", type=float, default=20.0)
    p.add_argument("--min-chunk-seconds", type=float, default=2.0)
    p.add_argument("--time-expansion-col", help="manifest column with the time-expansion factor of each "
                   "recording (1 = real time, will be expanded; equal to --time-expansion-factor = already expanded)")
    p.add_argument("--time-expansion-factor", type=float, default=10.0)
    p.add_argument("--expand-all", action="store_true", help="--input-dir only: time-expand every file")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()

    rows = rows_from_manifest(args) if args.manifest else rows_from_folders(args)
    log.info(f"{len(rows)} recordings -> {args.output_dir}")
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process, r, args) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            records.extend(fut.result())
            if i % 500 == 0:
                log.info(f"{i}/{len(rows)} recordings done, {len(records)} chunks")
    if not records:
        raise SystemExit("No chunks produced.")
    out = pd.DataFrame(records)
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_manifest, index=False)
    log.info(f"{len(out)} chunks written. Manifest: {args.output_manifest}")
    if "split" in out:
        log.info("Chunks per split:\n" + out["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
