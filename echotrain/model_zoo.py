"""Build a model by name from the unchanged legacy/models.py.

To make another architecture from legacy/models.py trainable, add an entry to
MODELS. A model receives raw waveforms (batch, samples) and returns either
species logits, or a (species_logits, genus_logits) tuple when two_heads=True.
"""
import sys
import types
import warnings
from pathlib import Path

import torch

from .frontend import PerchFrontend

LEGACY_DIR = Path(__file__).resolve().parent.parent / "legacy"


def _legacy_models():
    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))
    try:
        import perchv2_pytorch  # noqa: F401
    except ImportError:
        # Only the ONNX Perch models need this package; let the rest import without it.
        stub = types.ModuleType("perchv2_pytorch")

        class PerchONNXBackbone:
            def __init__(self, *a, **k):
                raise ImportError("perchv2_pytorch is not installed (needed for ONNX Perch models)")
        stub.PerchONNXBackbone = PerchONNXBackbone
        sys.modules["perchv2_pytorch"] = stub
    with warnings.catch_warnings():  # legacy/models.py uses an old timm import path
        warnings.filterwarnings("ignore", message="Importing from timm.models.layers", category=FutureWarning)
        import models
    return models


def _frontend(cfg):
    f = cfg["frontend"]
    return PerchFrontend(sr=cfg["audio"]["sample_rate"], n_mels=f["n_mels"], fmin=f["fmin"],
                         fmax=f["fmax"], freqm=f["freqm"], timem=f["timem"])


def _chunked(m, cfg, n_species, n_genus):
    model = m.ChiroEchoModelCHUNKED(n_species, n_genus, mel=_frontend(cfg), pretrained=cfg["model"]["pretrained"])
    chunk = int(round(cfg["model"]["chunk_seconds"] * cfg["audio"]["sample_rate"]))
    clip = int(round(cfg["audio"]["clip_seconds"] * cfg["audio"]["sample_rate"]))
    if clip % chunk:
        raise ValueError(f"audio.clip_seconds ({cfg['audio']['clip_seconds']}) must be a multiple of "
                         f"model.chunk_seconds ({cfg['model']['chunk_seconds']})")
    model.chunk_samples = chunk
    model.aggregation = cfg["model"]["chunk_aggregation"]
    return model


def _perch_weights(cfg):
    if not cfg["model"]["perch_weights"]:
        print("WARNING: model.perch_weights is not set; the Perch backbone starts from random weights.")
    return cfg["model"]["perch_weights"]


# name: (builder, two_heads, description)
MODELS = {
    "effnet_b3_chunked": (
        _chunked, True,
        "EfficientNet-B3 (ImageNet) on each chunk_seconds piece of the clip; chunk logits are pooled. "
        "Species + genus heads."),
    "effnet_b3": (
        lambda m, cfg, ns, ng: m.ChiroEchoModel(ns, ng, mel=_frontend(cfg), pretrained=cfg["model"]["pretrained"]),
        True, "EfficientNet-B3 (ImageNet) on the whole clip. Species + genus heads."),
    "perch_effnet_b3": (
        lambda m, cfg, ns, ng: m.ChiroEchoPerchModel(ns, ng, mel=_frontend(cfg), perch_weights_path=_perch_weights(cfg)),
        True, "EfficientNet-B3 with Perch v2 weights (model.perch_weights). Species + genus heads."),
    "effnet_b3_single": (
        lambda m, cfg, ns, ng: m.EfficientNetBat(ns, mel=_frontend(cfg), pretrained=cfg["model"]["pretrained"]),
        False, "EfficientNet-B3 (ImageNet), species head only."),
    "perch_effnet_b3_single": (
        lambda m, cfg, ns, ng: m.EfficientNetPerch(ns, mel=_frontend(cfg), perch_weights_path=_perch_weights(cfg)),
        False, "EfficientNet-B3 with Perch v2 weights, species head only."),
}


def build_model(cfg, label_space):
    name = cfg["model"]["name"]
    if name not in MODELS:
        raise ValueError(f"Unknown model.name {name!r}. Available: {', '.join(MODELS)}")
    builder, two_heads, _ = MODELS[name]
    if two_heads and not label_space.has_genus:
        raise ValueError(f"Model {name!r} has a genus head: set taxonomy.genus_head: true, "
                         f"or pick a *_single model.")
    if not two_heads and label_space.has_genus:
        raise ValueError(f"Model {name!r} has no genus head: set taxonomy.genus_head: false, "
                         f"or pick a model with two heads.")
    n_genus = len(label_space.genus) if label_space.has_genus else 0
    model = builder(_legacy_models(), cfg, len(label_space.species), n_genus)
    if cfg["model"]["init_checkpoint"]:
        warm_start(model, cfg["model"]["init_checkpoint"])
    return model


def warm_start(model, path):
    """Load every tensor whose name and shape match; the rest keeps its fresh init.

    Lets you start from a model trained on a different class list: the backbone
    is reused and only the output heads are re-learned.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]  # a last.pt checkpoint
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(set(state) - set(matched))
    model.load_state_dict(matched, strict=False)
    fresh = sorted(set(own) - set(matched))
    print(f"Warm start from {path}: loaded {len(matched)}/{len(own)} tensors.")
    if skipped:
        print(f"  not loaded (missing in model or other shape): {skipped}")
    if fresh:
        print(f"  freshly initialised: {fresh}")
