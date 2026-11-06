"""One training epoch, and evaluation with chunk -> recording pooling."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from .metrics import pairwise_confusion, summarize, top_confusions


def _split_outputs(out):
    if isinstance(out, (tuple, list)):
        return out[0], out[1]
    return out, None


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, has_genus):
    model.train()
    t = cfg["train"]
    criterion = nn.BCEWithLogitsLoss()
    accum = max(1, int(t["grad_accum_steps"]))
    running, n_batches = 0.0, len(loader)
    optimizer.zero_grad(set_to_none=True)

    bar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    for step, (inputs, y, g) in enumerate(bar):
        x = inputs["audio"].to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=t["amp"] and device.type == "cuda"):
            species_logits, genus_logits = _split_outputs(model(x))
            loss = criterion(species_logits.float(), y)
            if has_genus:
                loss = loss + t["genus_loss_weight"] * criterion(genus_logits.float(), g.to(device))
        scaler.scale(loss / accum).backward()

        if (step + 1) % accum == 0 or step + 1 == n_batches:
            if t["grad_clip"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running += loss.item()
        bar.set_postfix(loss=f"{running / (step + 1):.4f}")
    return running / max(n_batches, 1)


@torch.no_grad()
def predict(model, loader, device, has_genus):
    """Sigmoid scores per file, in dataset order."""
    model.eval()
    n = len(loader.dataset)
    species = genus = None
    loss_sum, criterion = 0.0, nn.BCEWithLogitsLoss()
    for inputs, y, g in tqdm(loader, desc="eval", leave=False, dynamic_ncols=True):
        idx = inputs["index"]
        s_logits, g_logits = _split_outputs(model(inputs["audio"].to(device)))
        loss = criterion(s_logits.float().cpu(), y)
        if has_genus:
            loss = loss + criterion(g_logits.float().cpu(), g)
        loss_sum += loss.item()
        if species is None:
            species = torch.zeros(n, s_logits.shape[1])
            if has_genus:
                genus = torch.zeros(n, g_logits.shape[1])
        species[idx] = torch.sigmoid(s_logits.float()).cpu()
        if has_genus:
            genus[idx] = torch.sigmoid(g_logits.float()).cpu()
    return species, genus, loss_sum / max(len(loader), 1)


def pool_by_group(values, groups):
    """Max over the files (chunks) of each recording. Returns (pooled, group_names)."""
    names, inverse = np.unique(np.asarray(groups), return_inverse=True)
    inverse = torch.from_numpy(inverse).long()
    out = torch.full((len(names), values.shape[1]), float("-inf"))
    out = out.scatter_reduce(0, inverse[:, None].expand_as(values), values, reduce="amax")
    return out, names


def _targets(idx_list, n):
    t = torch.zeros(len(idx_list), n)
    t[torch.arange(len(idx_list)), torch.as_tensor(idx_list)] = 1.0
    return t


def evaluate(model, loader, device, label_space, cfg):
    """Returns (metrics dict, reports dict of DataFrames/arrays to save)."""
    split = loader.dataset.split
    thr = cfg["eval"]["threshold"]
    has_genus = label_space.has_genus
    sp_probs, g_probs, loss = predict(model, loader, device, has_genus)

    sp_pooled, rec_names = pool_by_group(sp_probs, split.groups)
    sp_true, _ = pool_by_group(_targets(split.species_idx, len(label_space.species)), split.groups)
    metrics, ap = summarize(sp_pooled, sp_true, thr)
    metrics = {"loss": loss, **metrics, "recordings": len(rec_names)}

    conf = pairwise_confusion(sp_pooled, sp_true, thr)
    reports = {
        "per_class_species": pd.DataFrame({
            "class": label_space.species, "AP": ap,
            "n_recordings": sp_true.sum(0).long().numpy()}),
        "top_confusions": pd.DataFrame(
            top_confusions(conf, label_space.species, cfg["eval"]["top_n_confusions"]),
            columns=["true_class", "also_predicted", "recordings"]),
        "confusion": conf,
        "predictions": pd.DataFrame(sp_pooled.numpy(), columns=label_space.species).assign(
            recording=rec_names,
            true_class=[", ".join(label_space.species[j] for j in np.flatnonzero(r)) for r in sp_true.numpy()],
        ),
    }

    if has_genus:
        g_pooled, _ = pool_by_group(g_probs, split.groups)
        g_true, _ = pool_by_group(_targets(split.genus_idx, len(label_space.genus)), split.groups)
        g_metrics, g_ap = summarize(g_pooled, g_true, thr)
        metrics.update({f"genus_{k}": v for k, v in g_metrics.items()})
        reports["per_class_genus"] = pd.DataFrame({
            "genus": label_space.genus, "AP": g_ap, "n_recordings": g_true.sum(0).long().numpy()})
    return metrics, reports


def save_reports(reports, out_dir, label_space):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in reports.items():
        if isinstance(obj, pd.DataFrame):
            cols = None
            if name == "predictions":  # recording + truth first, then one score column per class
                cols = ["recording", "true_class"] + label_space.species
            obj.to_csv(out_dir / f"{name}.csv", index=False, columns=cols)
    np.savez(out_dir / "confusion.npz", pairwise=reports["confusion"], class_names=np.array(label_space.species))
