"""Train a species classifier.

    uv run train.py --config configs/bats.yaml
    uv run train.py --config configs/bats.yaml --set train.batch_size=8 --set run_name=bs8
    uv run train.py --resume runs/bats_effnet_b3_chunked/last.pt

Everything a run produces goes to <output_dir>/<run_name>/:
    config.yaml        the full config used (all defaults filled in)
    labels.json        class names in model output order (needed for inference)
    best_model.pt      weights (state_dict) of the epoch with the best validation mAP
    last.pt            weights + optimizer state of the latest epoch, for --resume
    history.csv        one row of metrics per epoch
    best_val/          per-class AP, top confusions and recording-level predictions of the best epoch
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from echotrain.data import LabelSpace, build_label_space, describe, encode, read_split
from echotrain.dataset import AudioDataset
from echotrain.engine import evaluate, save_reports, train_one_epoch
from echotrain.model_zoo import build_model
from echotrain.options import load_config, save_config
from echotrain.sampler import balanced_sampler


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    # Different numpy RNG state per worker, so augmentation differs between workers.
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


def make_loaders(cfg, train_split, val_split, label_space):
    n_sp = len(label_space.species)
    n_g = len(label_space.genus) if label_space.has_genus else 0
    train_set = AudioDataset(train_split, n_sp, n_g, cfg, train=True)
    val_set = AudioDataset(val_split, n_sp, n_g, cfg, train=False)

    s = cfg["sampler"]
    sampler = (balanced_sampler(train_split.labels, s["cap"], s["power"], s["noise_class"], s["noise_multiplier"])
               if s["balanced"] else None)
    t, e = cfg["train"], cfg["eval"]
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=t["batch_size"], sampler=sampler, shuffle=sampler is None,
        num_workers=t["num_workers"], pin_memory=True, drop_last=True, worker_init_fn=worker_init,
        persistent_workers=t["num_workers"] > 0)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=e["batch_size"], shuffle=False, num_workers=e["num_workers"],
        pin_memory=True, persistent_workers=e["num_workers"] > 0)
    return train_loader, val_loader


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML config file")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set train.lr=3e-4 (repeatable)")
    p.add_argument("--resume", help="continue a run from its last.pt (uses that run's config.yaml)")
    args = p.parse_args()

    if args.resume:
        run_dir = Path(args.resume).parent
        cfg = load_config(run_dir / "config.yaml", args.set)
    elif args.config:
        cfg = load_config(args.config, args.set)
        run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
        if (run_dir / "last.pt").exists():
            raise SystemExit(f"{run_dir} already holds a run. Pick another run_name, or use --resume {run_dir}/last.pt")
    else:
        p.error("give --config or --resume")

    seed_everything(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Run directory: {run_dir}  |  device: {device}")

    # ---- data -------------------------------------------------------------
    train_split = read_split(cfg, "train")
    val_split = read_split(cfg, "val")
    if args.resume:
        label_space = LabelSpace.load(run_dir / "labels.json")
    else:
        label_space = build_label_space(cfg, train_split)
    train_split = encode(train_split, label_space, "train")
    val_split = encode(val_split, label_space, "val")
    describe(train_split, "train")
    describe(val_split, "val")
    missing = sorted(set(label_space.species) - set(train_split.labels))
    if missing:
        print(f"WARNING: {len(missing)} classes have no training files: {missing}")
    print(f"{len(label_space.species)} classes" +
          (f", {len(label_space.genus)} genera" if label_space.has_genus else ""))

    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    label_space.save(run_dir / "labels.json")

    train_loader, val_loader = make_loaders(cfg, train_split, val_split, label_space)

    # ---- model and optimisation --------------------------------------------
    model = build_model(cfg, label_space).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model {cfg['model']['name']}: {n_params / 1e6:.1f}M trainable parameters")

    t = cfg["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = None
    if t["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=t["plateau_factor"], patience=t["plateau_patience"], min_lr=t["min_lr"])
    elif t["scheduler"] != "none":
        raise ValueError("train.scheduler must be 'plateau' or 'none'")
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"] and device.type == "cuda")

    start_epoch, best_map, best_epoch, history = 0, -1.0, -1, []
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if scheduler and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_map, best_epoch = ckpt["epoch"] + 1, ckpt["best_map"], ckpt["best_epoch"]
        if (run_dir / "history.csv").exists():
            history = pd.read_csv(run_dir / "history.csv").to_dict("records")[:start_epoch]
        print(f"Resuming at epoch {start_epoch + 1} (best mAP so far {best_map:.4f} at epoch {best_epoch + 1})")

    # ---- loop -------------------------------------------------------------
    for epoch in range(start_epoch, t["epochs"]):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, label_space.has_genus)
        metrics, reports = evaluate(model, val_loader, device, label_space, cfg)
        if scheduler:
            scheduler.step(metrics["mAP"])

        improved = metrics["mAP"] > best_map
        if improved:
            best_map, best_epoch = metrics["mAP"], epoch
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            save_reports(reports, run_dir / "best_val", label_space)
            with open(run_dir / "best_val" / "metrics.json", "w") as f:
                json.dump({"epoch": epoch + 1, **metrics}, f, indent=2)

        row = {"epoch": epoch + 1, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"],
               **{f"val_{k}": v for k, v in metrics.items()}, "minutes": (time.time() - t0) / 60}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None, "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_map": best_map, "best_epoch": best_epoch}, run_dir / "last.pt")

        msg = (f"Epoch {epoch + 1}/{t['epochs']}  train_loss {train_loss:.4f}  val_loss {metrics['loss']:.4f}  "
               f"mAP {metrics['mAP']:.4f}  F1 {metrics['f1']:.4f}  AUROC {metrics['auroc']:.4f}")
        if label_space.has_genus:
            msg += f"  | genus mAP {metrics['genus_mAP']:.4f}  F1 {metrics['genus_f1']:.4f}"
        print(msg + ("  * best" if improved else f"  (best {best_map:.4f} @ {best_epoch + 1})"), flush=True)

        patience = t["early_stop_patience"]
        if patience and epoch - best_epoch >= patience:
            print(f"No improvement for {patience} epochs, stopping.")
            break

    print(f"Done. Best validation mAP {best_map:.4f} at epoch {best_epoch + 1}: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
