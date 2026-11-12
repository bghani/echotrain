"""Score a trained run on another split (e.g. the test set).

    # manifest datasets: pick split values from the manifest
    uv run evaluate.py --run runs/bats_effnet_b3 --split test
    # folder datasets: point at a folder with one subfolder per class
    uv run evaluate.py --run runs/my_birds --dir /data/my_birds/test

Writes per-class AP, top confusions, recording-level predictions and
metrics.json to <run>/eval_<name>/.
"""
import argparse
import json
from pathlib import Path

import torch

from echotrain.data import LabelSpace, describe, encode, read_split
from echotrain.dataset import AudioDataset
from echotrain.device import pick_device
from echotrain.engine import evaluate, save_reports
from echotrain.model_zoo import build_model
from echotrain.options import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory written by train.py")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", nargs="+", help="manifest split value(s) to evaluate")
    g.add_argument("--dir", help="folder with one subfolder per class")
    p.add_argument("--checkpoint", help="weights to use (default: <run>/best_model.pt)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    args = p.parse_args()

    run = Path(args.run)
    cfg = load_config(run / "config.yaml", args.set)
    cfg["model"]["init_checkpoint"] = None
    cfg["model"]["pretrained"] = False   # weights come from the checkpoint
    label_space = LabelSpace.load(run / "labels.json")

    if args.dir:
        cfg["data"]["mode"] = "folders"
        split, name = read_split(cfg, args.dir), Path(args.dir).name
    else:
        split, name = read_split(cfg, args.split), "_".join(args.split)
    split = encode(split, label_space, name)
    describe(split, name)

    device = pick_device(cfg["device"])
    model = build_model(cfg, label_space)
    model.load_state_dict(torch.load(args.checkpoint or run / "best_model.pt", map_location="cpu", weights_only=True))
    model.to(device)

    n_g = len(label_space.genus) if label_space.has_genus else 0
    loader = torch.utils.data.DataLoader(
        AudioDataset(split, len(label_space.species), n_g, cfg, train=False),
        batch_size=cfg["eval"]["batch_size"], shuffle=False, num_workers=cfg["eval"]["num_workers"])
    metrics, reports = evaluate(model, loader, device, label_space, cfg)

    out = run / f"eval_{name}"
    save_reports(reports, out, label_space)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Reports written to {out}")


if __name__ == "__main__":
    main()
