"""Build a model by name.

The EfficientNet and PaSST models live in echotrain/networks.py; the Perch
models come from the unchanged legacy/models.py. To make another architecture
trainable, add an entry to MODELS. A model receives raw waveforms
(batch, samples) and returns either species logits, or a
(species_logits, genus_logits) tuple when two_heads=True.
"""
import sys
import types
import warnings
from pathlib import Path

import torch

from . import networks
from .frontend import PasstFrontend, PerchFrontend

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


def _effnet(cfg):
    name, pretrained = cfg["model"]["backbone"], cfg["model"]["pretrained"]
    networks.check_efficientnet(name, pretrained)
    return name, pretrained


def _chunked(m, cfg, n_species, n_genus):
    chunk = int(round(cfg["model"]["chunk_seconds"] * cfg["audio"]["sample_rate"]))
    clip = int(round(cfg["audio"]["clip_seconds"] * cfg["audio"]["sample_rate"]))
    if clip % chunk:
        raise ValueError(f"audio.clip_seconds ({cfg['audio']['clip_seconds']}) must be a multiple of "
                         f"model.chunk_seconds ({cfg['model']['chunk_seconds']})")
    return networks.EffNetChunked(_effnet(cfg)[0], n_species, n_genus, _frontend(cfg),
                                  pretrained=cfg["model"]["pretrained"], chunk_samples=chunk,
                                  aggregation=cfg["model"]["chunk_aggregation"])


def _whole(m, cfg, n_species, n_genus):
    name, pretrained = _effnet(cfg)
    return networks.EffNet(name, n_species, n_genus, _frontend(cfg), pretrained=pretrained)


def _single(m, cfg, n_species, n_genus):
    name, pretrained = _effnet(cfg)
    return networks.EffNetSingle(name, n_species, _frontend(cfg), pretrained=pretrained)


def _passt(m, cfg, n_species, n_genus):
    mc, f = cfg["model"], cfg["frontend"]
    mel = PasstFrontend(sr=cfg["audio"]["sample_rate"], n_mels=f["n_mels"], fmin=mc["passt_fmin"],
                        fmax=mc["passt_fmax"], freqm=f["freqm"], timem=f["timem"])
    return networks.Passt(mc["passt_arch"], n_species, n_genus, mel, pretrained=mc["pretrained"],
                          s_patchout_t=mc["passt_patchout_t"], s_patchout_f=mc["passt_patchout_f"])


def _perch_weights(cfg):
    if not cfg["model"]["perch_weights"]:
        print("WARNING: model.perch_weights is not set; the Perch backbone starts from random weights.")
    return cfg["model"]["perch_weights"]


# name: (builder, two_heads, needs_legacy, description)
MODELS = {
    "effnet_chunked": (
        _chunked, True, False,
        "EfficientNet (model.backbone, ImageNet) on each chunk_seconds piece of the clip; chunk logits are "
        "pooled. Species + genus heads."),
    "effnet": (_whole, True, False, "EfficientNet (model.backbone, ImageNet) on the whole clip. Species + genus heads."),
    "effnet_single": (_single, False, False, "EfficientNet (model.backbone, ImageNet), species head only."),
    "passt": (
        _passt, True, False,
        "PaSST transformer (model.passt_arch, AudioSet) on the whole clip. Species + genus heads."),
    "passt_single": (_passt, False, False, "PaSST transformer (model.passt_arch, AudioSet), species head only."),
    "perch_effnet_b3": (
        lambda m, cfg, ns, ng: m.ChiroEchoPerchModel(ns, ng, mel=_frontend(cfg), perch_weights_path=_perch_weights(cfg)),
        True, True, "EfficientNet-B3 with Perch v2 weights (model.perch_weights). Species + genus heads."),
    "perch_effnet_b3_single": (
        lambda m, cfg, ns, ng: m.EfficientNetPerch(ns, mel=_frontend(cfg), perch_weights_path=_perch_weights(cfg)),
        False, True, "EfficientNet-B3 with Perch v2 weights, species head only."),
}

# Names used before model.backbone existed. They build the same model as the
# new name; the backbone comes from model.backbone (default efficientnet_b3).
ALIASES = {
    "effnet_b3_chunked": "effnet_chunked",
    "effnet_b3": "effnet",
    "effnet_b3_single": "effnet_single",
}


def build_model(cfg, label_space):
    name = cfg["model"]["name"]
    if name in ALIASES:
        if cfg["model"]["backbone"] != "efficientnet_b3":
            raise ValueError(f"model.name {name!r} is EfficientNet-B3 only; use {ALIASES[name]!r} "
                             f"with model.backbone={cfg['model']['backbone']!r}")
        name = ALIASES[name]
    if name not in MODELS:
        raise ValueError(f"Unknown model.name {name!r}. Available: {', '.join(MODELS)}")
    builder, two_heads, needs_legacy, _ = MODELS[name]
    if two_heads and not label_space.has_genus:
        raise ValueError(f"Model {name!r} has a genus head: set taxonomy.genus_head: true, "
                         f"or pick a *_single model.")
    if not two_heads and label_space.has_genus:
        raise ValueError(f"Model {name!r} has no genus head: set taxonomy.genus_head: false, "
                         f"or pick a model with two heads.")
    n_genus = len(label_space.genus) if label_space.has_genus else 0
    model = builder(_legacy_models() if needs_legacy else None, cfg, len(label_space.species), n_genus)
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
