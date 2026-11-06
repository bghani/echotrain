"""Append a background/noise class to a chunk manifest.

Every audio file under <noise-dir>/train, <noise-dir>/val and <noise-dir>/test
becomes one row with label --label and the split taken from its subfolder.
Other columns are left empty. Paths are written relative to --audio-root (the
same root used for training), or absolute if --audio-root is not given.

    uv run tools/add_noise_to_manifest.py --manifest manifest_chunks.csv \\
        --noise-dir /data/shared/data/noise --audio-root /data/shared/data \\
        --output manifest_chunks_with_noise.csv

Then set sampler.noise_class to the same label to down-weight it while training.
"""
import argparse
import os
from pathlib import Path

import pandas as pd

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--noise-dir", required=True, help="folder with train/ val/ test/ subfolders")
    p.add_argument("--output", required=True)
    p.add_argument("--audio-root", help="write paths relative to this folder")
    p.add_argument("--label", default="noise")
    p.add_argument("--label-col", default="latin")
    p.add_argument("--path-col", default="chunk_path")
    p.add_argument("--split-col", default="split")
    p.add_argument("--group-col", default="file_path",
                   help="column that names the source recording; each noise file is its own recording")
    args = p.parse_args()

    df = pd.read_csv(args.manifest, low_memory=False)
    rows = []
    for split in ("train", "val", "test"):
        folder = Path(args.noise_dir) / split
        if not folder.is_dir():
            print(f"skipping missing folder {folder}")
            continue
        files = sorted(f for f in folder.rglob("*") if f.suffix.lower() in AUDIO_EXT)
        for f in files:
            path = str(f.resolve())
            rel = os.path.relpath(path, args.audio_root) if args.audio_root else path
            row = {args.label_col: args.label, args.split_col: split, args.path_col: rel}
            if args.group_col:
                row[args.group_col] = path
            rows.append(row)
        print(f"{split}: {len(files)} noise files")
    if not rows:
        raise SystemExit("No noise files found.")

    noise = pd.DataFrame(rows)
    out = pd.concat([df, noise], ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(df)} existing rows + {len(noise)} noise rows")


if __name__ == "__main__":
    main()
