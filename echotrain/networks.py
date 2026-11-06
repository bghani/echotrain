"""Model classes used by model_zoo.py.

The EfficientNet classes behave like ChiroEchoModel and EfficientNetBat in legacy/models.py, but take the timm backbone name as an
argument (efficientnet_b0 ... b5, tf_efficientnet_b0 ... b8, efficientnetv2,
...). Parameter names are the same as in the legacy classes, so checkpoints
of the legacy EfficientNet-B3 models load into the b3 versions of these.

Every model takes raw waveforms (batch, samples) and returns species logits,
or (species_logits, genus_logits) when it has a genus head.
"""
import sys
import warnings
from pathlib import Path

import timm
import torch.nn as nn

LEGACY_DIR = Path(__file__).resolve().parent.parent / "legacy"


def check_efficientnet(name, pretrained):
    """Fail early, with the list of valid names, if timm cannot build `name`."""
    available = timm.list_models("*efficientnet*")
    if name not in available:
        raise ValueError(f"model.backbone={name!r} is not an EfficientNet known to timm. "
                         f"Examples: efficientnet_b0 ... efficientnet_b5, tf_efficientnet_b0 ... tf_efficientnet_b8, "
                         f"tf_efficientnetv2_s. All: {', '.join(available)}")
    if pretrained and not timm.list_models(name, pretrained=True):
        with_weights = sorted({m.split(".")[0] for m in timm.list_models("*efficientnet*", pretrained=True)})
        raise ValueError(f"timm has no pretrained weights for {name!r}. Set model.pretrained: false, or pick one "
                         f"of: {', '.join(with_weights)}")


class EffNet(nn.Module):
    """EfficientNet on the whole clip, single-channel input. Species head, optional genus head."""

    def __init__(self, backbone, num_species, num_genus, mel, pretrained=True):
        super().__init__()
        self.mel = mel
        self.backbone = timm.create_model(backbone, in_chans=1, pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.num_features
        self.species_head = nn.Linear(embed_dim, num_species)
        self.genus_head = nn.Linear(embed_dim, num_genus) if num_genus else None

    def heads(self, emb):
        if self.genus_head is None:
            return self.species_head(emb)
        return self.species_head(emb), self.genus_head(emb)

    def forward(self, x):
        return self.heads(self.backbone(self.mel(x).unsqueeze(1)))


class EffNetSingle(nn.Module):
    """Same as legacy EfficientNetBat: spectrogram repeated to 3 channels, timm classifier as species head."""

    def __init__(self, backbone, num_species, mel, pretrained=True):
        super().__init__()
        self.mel = mel
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=num_species)

    def forward(self, x):
        return self.backbone(self.mel(x).unsqueeze(1).repeat(1, 3, 1, 1))


# PaSST architectures from hear21passt and the number of spectrogram frames
# (at a 10 ms hop) their time position encodings were trained for.
PASST_ARCHS = {
    "passt_s_kd_p16_128_ap486": 998,     # 10 s, AudioSet, knowledge distillation (mAP .486)
    "passt_s_swa_p16_128_ap476": 998,    # 10 s, AudioSet
    "passt_l_kd_p16_128_ap47": 998,      # 10 s, PaSST-L (light)
    "passt_20sec": 2000,                 # 20 s, AudioSet
    "passt_30sec": 3000,                 # 30 s, AudioSet
    "openmic": 998,
    "fsd50k": 998,
}


def _passt_module():
    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))
    with warnings.catch_warnings():  # the copied code uses old timm import paths
        warnings.filterwarnings("ignore", category=FutureWarning, message="Importing from timm")
        from hear21passt.models import passt
    return passt


class Passt(nn.Module):
    """PaSST transformer (legacy/hear21passt) on the whole clip. Species head, optional genus head.

    The species head is PaSST's own head (net.head); the genus head is a
    LayerNorm + Linear on the same 768-d clip embedding. Parameters are named
    like the old PasstBasicWrapper (net.*), so old batecho/insectaecho PaSST
    checkpoints can be used as model.init_checkpoint.
    """

    def __init__(self, arch, num_species, num_genus, mel, pretrained=True, s_patchout_t=0, s_patchout_f=0):
        super().__init__()
        if arch not in PASST_ARCHS:
            raise ValueError(f"model.passt_arch={arch!r} is unknown. Available: {', '.join(PASST_ARCHS)}")
        self.mel = mel
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Removing representation layer")
            self.net = _passt_module().get_model(
                arch=arch, pretrained=pretrained, n_classes=num_species, in_channels=1,
                input_fdim=mel.n_mels, input_tdim=PASST_ARCHS[arch],
                s_patchout_t=s_patchout_t, s_patchout_f=s_patchout_f)
        dim = self.net.num_features
        self.genus_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_genus)) if num_genus else None

    def forward(self, x):
        spec = self.mel(x).unsqueeze(1)  # (batch, 1, n_mels, time)
        with warnings.catch_warnings():  # clip length differs from the pretraining length: expected, handled
            warnings.filterwarnings("ignore", message="Input image size")
            logits, emb = self.net(spec)
        if self.genus_head is None:
            return logits
        return logits, self.genus_head(emb)
