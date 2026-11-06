"""Scores computed on recording-level (pooled) predictions."""
import numpy as np
import torch
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score


def average_precision_per_class(scores, labels):
    """Average precision per class (non-interpolated). Classes without positives get NaN."""
    aps = []
    for c in range(scores.shape[1]):
        y = labels[:, c]
        n_pos = y.sum()
        if n_pos == 0:
            aps.append(float("nan"))
            continue
        y_sorted = y[torch.argsort(scores[:, c], descending=True)]
        precision = torch.cumsum(y_sorted, 0) / torch.arange(1, len(y_sorted) + 1)
        aps.append(float((precision * y_sorted).sum() / n_pos))
    return np.array(aps)


def summarize(probs, labels, threshold):
    """mAP over classes present in the labels, plus weighted F1 and AUROC."""
    n = labels.shape[1]
    ap = average_precision_per_class(probs, labels)
    present = ~np.isnan(ap)
    y = labels.long()
    out = {
        "mAP": float(np.nanmean(ap)) if present.any() else float("nan"),
        "f1": float(MultilabelF1Score(num_labels=n, average="weighted", threshold=threshold)(probs, y)),
        "classes_scored": int(present.sum()),
    }
    try:
        out["auroc"] = float(MultilabelAUROC(num_labels=n, average="weighted")(probs, y))
    except Exception:
        out["auroc"] = float("nan")
    return out, ap


def pairwise_confusion(probs, labels, threshold):
    """C[i, j] (i != j): recordings of true class i where class j fired but is not present.
    C[i, i]: true positives of class i."""
    pred = probs >= threshold
    true = labels.bool()
    false_pos = pred & ~true
    conf = (true.T.float() @ false_pos.float()).long()
    tp = (pred & true).sum(0)
    conf[torch.arange(len(tp)), torch.arange(len(tp))] = tp
    return conf.numpy()


def top_confusions(conf, names, n):
    pairs = [(names[i], names[j], int(conf[i, j]))
             for i in range(len(names)) for j in range(len(names)) if i != j and conf[i, j] > 0]
    return sorted(pairs, key=lambda p: p[2], reverse=True)[:n]
