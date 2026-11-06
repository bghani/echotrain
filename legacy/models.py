import torch
import torch.nn as nn
#from efficientnet_pytorch import EfficientNet
import torchvision
from root_modules import *
from torch.nn import init
from config import *
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn.functional as F
import torchvision.models as models
#import wget
import torchaudio
import timm
from timm.models.layers import to_2tuple,trunc_normal_
from torch_audiomentations import AddColoredNoise, ApplyImpulseResponse, Shift, AddBackgroundNoise
from util import min_max_norm
from perchv2_pytorch import PerchONNXBackbone
#torch.autograd.set_detect_anomaly(True) 



import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientNetPerch(nn.Module):
    def __init__(self, num_classes: int, mel: nn.Module, perch_weights_path: str = None):
        super().__init__()
        self.mel = mel
        self.backbone = timm.create_model(
            'tf_efficientnet_b3',
            pretrained=False,
            in_chans=1,          # Perch's stem is single-channel, not RGB-style
            num_classes=num_classes,
        )
        if perch_weights_path:
            state = torch.load(perch_weights_path, map_location="cpu")
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            # missing should only be 'classifier.*' (replaced for your num_classes)
            print(f'Missing: {missing}')
            print(f'Unexpected: {unexpected}')

    def forward(self, x):
        x = self.mel(x)
        #x = x.transpose(1, 2)  
        x = x.unsqueeze(1)   # single channel -- no .repeat() here
        return self.backbone(x)



class EfficientNetBat(nn.Module):
    def __init__(self, num_classes: int, mel: nn.Module, pretrained: bool = True):
        super().__init__()
        self.mel = mel
        self.backbone = timm.create_model(
            'efficientnet_b3', #tf_efficientnetv2_s
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        #x = self.mel(x.float()) 
        x = self.mel(x)
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


class ChiroEchoPerchModel(nn.Module):
    def __init__(self, num_species: int, num_genus: int, onnx_path: str,
                 cache_dir: str = None, chunk_samples: int = 160_000,
                 aggregation: str = "mean"):
        super().__init__()
        self.backbone = PerchONNXBackbone(onnx_path, cache_dir=cache_dir)
        self.chunk_samples = chunk_samples
        self.aggregation = aggregation

        embed_dim = 1536
        self.species_head = nn.Linear(embed_dim, num_species)
        self.genus_head = nn.Linear(embed_dim, num_genus)

    def forward(self, x: torch.Tensor):
        # x: (batch, total_samples) -- total_samples must be an exact
        # multiple of chunk_samples (e.g. 640000 / 160000 = 4 chunks/clip).
        # All ops below are pure reshapes -- zero data movement, stays on
        # whatever device x is already on.
        batch_size, total_samples = x.shape
        if total_samples % self.chunk_samples != 0:
            raise ValueError(
                f"Input length {total_samples} is not an exact multiple of "
                f"chunk_samples={self.chunk_samples} -- pad/trim upstream, "
                f"not here."
            )
        n_chunks = total_samples // self.chunk_samples

        # (batch, total_samples) -> (batch*n_chunks, chunk_samples): consecutive,
        # non-overlapping 5s windows, grouped by original clip -- verified above.
        x = x.reshape(batch_size * n_chunks, self.chunk_samples)

        emb = self.backbone(x)  # (batch*n_chunks, 1536) -- one real forward call

        # Regroup by clip and aggregate the n_chunks embeddings back to one per clip
        emb = emb.view(batch_size, n_chunks, -1)
        if self.aggregation == "mean":
            emb = emb.mean(dim=1)
        elif self.aggregation == "max":
            emb = emb.amax(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation!r}")

        return self.species_head(emb), self.genus_head(emb)


class ChiroEchoPerch2ONNXModel(nn.Module):
    def __init__(self, num_species: int, num_genus: int, onnx_path: str,
                 cache_dir: str = None, chunk_samples: int = 160_000,
                 aggregation: str = "max"):
        super().__init__()
        self.backbone = PerchONNXBackbone(onnx_path, cache_dir=cache_dir)
        self.chunk_samples = chunk_samples
        self.aggregation = aggregation

        embed_dim = 1536
        self.species_head = nn.Linear(embed_dim, num_species)
        self.genus_head = nn.Linear(embed_dim, num_genus)

    def forward(self, x: torch.Tensor):
        batch_size, total_samples = x.shape
        if total_samples % self.chunk_samples != 0:
            raise ValueError(
                f"Input length {total_samples} is not an exact multiple of "
                f"chunk_samples={self.chunk_samples} -- pad/trim upstream."
            )
        n_chunks = total_samples // self.chunk_samples

        x = x.reshape(batch_size * n_chunks, self.chunk_samples)
        #with torch.cuda.amp.autocast(enabled=False):
        emb = self.backbone(x)  # (batch*n_chunks, 1536)
        

        # Apply the SAME shared head to every chunk independently -- each
        # chunk gets its own species/genus logits before any aggregation.
        species_logits = self.species_head(emb)  # (batch*n_chunks, num_species)
        genus_logits = self.genus_head(emb)      # (batch*n_chunks, num_genus)

        # Aggregate LOGITS (not embeddings) back to one prediction per clip.
        # Logits are what BCEWithLogitsLoss directly trains to mean "higher
        # = more evidence this class is present" -- embeddings have no such
        # guaranteed property on any dimension, trained or not.
        species_logits = species_logits.view(batch_size, n_chunks, -1)
        genus_logits = genus_logits.view(batch_size, n_chunks, -1)

        if self.aggregation == "mean":
            species_logits = species_logits.mean(dim=1)
            genus_logits = genus_logits.mean(dim=1)
        elif self.aggregation == "max":
            species_logits = species_logits.amax(dim=1)
            genus_logits = genus_logits.amax(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation!r}")

        return species_logits, genus_logits


class ChiroEchoPerch2ONNXModelOLD(nn.Module):
    def __init__(self, num_species, num_genus, onnx_path, cache_dir=None):
        super().__init__()
        self.backbone = PerchONNXBackbone(onnx_path, cache_dir=cache_dir)
        self.species_head = nn.Linear(1536, num_species)
        self.genus_head = nn.Linear(1536, num_genus)

    def forward(self, waveform):
        #with torch.cuda.amp.autocast(enabled=False):
        emb = self.backbone(waveform[:,:160000])  # raw waveform in, no separate mel step
        return self.species_head(emb), self.genus_head(emb)

class ChiroEchoPerchModel(nn.Module):
    def __init__(self, num_species: int, num_genus: int, mel: nn.Module, perch_weights_path: str = None):
        super().__init__()
        self.mel = mel
        self.backbone = timm.create_model(
            'tf_efficientnet_b3',
            pretrained=False,
            in_chans=1,          # Perch's stem is single-channel, not RGB-style
            num_classes=0,       # strip classifier, return pooled embedding
        )
        if perch_weights_path:
            state = torch.load(perch_weights_path, map_location="cpu")
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            # missing should now include 'classifier.*' since we removed it entirely (num_classes=0)
            print(f'Missing: {missing}')
            print(f'Unexpected: {unexpected}')

        embed_dim = self.backbone.num_features
        self.species_head = nn.Linear(embed_dim, num_species)
        self.genus_head = nn.Linear(embed_dim, num_genus)

    def forward(self, x):
        x = self.mel(x)
        x = x.unsqueeze(1)   # single channel -- no .repeat() here
        emb = self.backbone(x)
        return self.species_head(emb), self.genus_head(emb)

class ChiroEchoModel(nn.Module):
    def __init__(self, num_species: int, num_genus: int, mel: nn.Module, pretrained: bool = True):
        super().__init__()
        self.mel = mel
        self.backbone = timm.create_model(
            'efficientnet_b3',
            in_chans=1,
            pretrained=pretrained,
            num_classes=0,  # strips the classifier, returns pooled embedding
        )
        embed_dim = self.backbone.num_features  # 1536 for efficientnet_b3
        self.species_head = nn.Linear(embed_dim, num_species)
        self.genus_head = nn.Linear(embed_dim, num_genus)

    def forward(self, x):
        x = self.mel(x)
        x = x.unsqueeze(1)
        #x = x.repeat(1, 3, 1, 1)
        emb = self.backbone(x)  # (batch, embed_dim)
        return self.species_head(emb), self.genus_head(emb)        

class ChiroEchoModelCHUNKED(nn.Module):
    def __init__(self, num_species: int, num_genus: int, mel: nn.Module, pretrained: bool = True):
        super().__init__()
        self.mel = mel
        self.backbone = timm.create_model(
            'efficientnet_b3',
            in_chans=1,
            pretrained=pretrained,
            num_classes=0,  # strips the classifier, returns pooled embedding
        )
        embed_dim = self.backbone.num_features  # 1536 for efficientnet_b3
        self.species_head = nn.Linear(embed_dim, num_species)
        self.genus_head = nn.Linear(embed_dim, num_genus)
        self.chunk_samples = 160000
        self.aggregation = 'max'

    def forward(self, x):
        batch_size, total_samples = x.shape
        if total_samples % self.chunk_samples != 0:
            raise ValueError(
                f"Input length {total_samples} is not an exact multiple of "
                f"chunk_samples={self.chunk_samples} -- pad/trim upstream."
            )
        n_chunks = total_samples // self.chunk_samples
        x = x.reshape(batch_size * n_chunks, self.chunk_samples)
        x = self.mel(x)
        x = x.unsqueeze(1)
        #x = x.repeat(1, 3, 1, 1)
        emb = self.backbone(x)  # (batch, embed_dim)

        # Apply the SAME shared head to every chunk independently -- each
        # chunk gets its own species/genus logits before any aggregation.
        species_logits = self.species_head(emb)  # (batch*n_chunks, num_species)
        genus_logits = self.genus_head(emb)      # (batch*n_chunks, num_genus)

        # Aggregate LOGITS (not embeddings) back to one prediction per clip.
        # Logits are what BCEWithLogitsLoss directly trains to mean "higher
        # = more evidence this class is present" -- embeddings have no such
        # guaranteed property on any dimension, trained or not.
        species_logits = species_logits.view(batch_size, n_chunks, -1)
        genus_logits = genus_logits.view(batch_size, n_chunks, -1)

        if self.aggregation == "mean":
            species_logits = species_logits.mean(dim=1)
            genus_logits = genus_logits.mean(dim=1)
        elif self.aggregation == "max":
            species_logits = species_logits.amax(dim=1)
            genus_logits = genus_logits.amax(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation!r}")

        return species_logits, genus_logits
        #return self.species_head(emb), self.genus_head(emb)    



class BatShallowCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        target_freq_bins: int = 115,
        target_time_frames: int = 40,
        log_compression: bool = True,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.target_freq_bins = target_freq_bins
        self.target_time_frames = target_time_frames
        self.log_compression = log_compression

        # Window buffer for STFT
        self.register_buffer("window", torch.hann_window(win_length))

        # Shallow CNN
        self.features = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),

        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),

        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),

        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

           
    def waveform_to_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T]
        returns: [B, 1, target_freq_bins, target_time_frames]
        """
        window = self.window.to(x.device)
        # STFT: [B, F, TT] complex
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )

        # Magnitude spectrogram: [B, F, TT]
        spec = spec.abs()

        # Optional log compression
        if self.log_compression:
            spec = torch.log1p(spec)

        # Add channel dim: [B, 1, F, TT]
        spec = spec.unsqueeze(1)

        # Resize to fixed size [115, 40]
        spec = F.interpolate(
            spec,
            size=(self.target_freq_bins, self.target_time_frames),
            mode="bilinear",
            align_corners=False,
        )

        return spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 25600]
        returns: [B, num_classes]
        """
        spec = self.waveform_to_spectrogram(x)
        feats = self.features(spec)
        feats = feats.flatten(1)
        logits = self.classifier(feats)
        return logits


class SpectrogramCNN(nn.Module):
    def __init__(self, cfg, init_backbone=True):
        """
        Pytorch network class containing the transformation from waveform to
        mel spectrogram, as well as the forward pass through a CNN backbone.

        Data augmentation like mixup or masked frequency or time can also be
        applied here.

        Parameters
        ----------
        cfg: SimpleNameSpace containing all configurations
        init_backbone: bool (Default=True). Whether to download and initialize the backbone.
                       Not always necessary when debugging.
        """
        super(SpectrogramCNN, self).__init__()

        self.cfg = cfg
        self.n_classes = cfg.n_classes

        # Initializes the transformation from waveform to mel spectrogram
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.window_size,
            hop_length=cfg.hop_length,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            n_mels=cfg.n_mels,
            power=cfg.power,
            normalized=cfg.mel_normalized,
        )
        self.impulse = ApplyImpulseResponse(p=cfg.impulse_prob, p_mode="per_example",
                                            sample_rate=cfg.sample_rate, mode="per_example",
                                            compensate_for_propagation_delay=True,
                                            ir_paths='/data/burooj/models/insectaecho/audio/irs/examples')
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=cfg.top_db)
        self.wav2timefreq = torch.nn.Sequential(self.mel_spec,
                                           self.amplitude_to_db)

        if init_backbone:
            # Initialize pre-trained CNN
            # Input and output layers are automatically adjusted
            self.backbone = timm.create_model(
                cfg.backbone,
                pretrained=cfg.pretrained,
                num_classes=cfg.n_classes,
                in_chans=cfg.in_chans,
            )

        # Spectrogram augmentation
        # Chooses one of MaskFrequency and MaskTime with a probability of cfg.specaug_prob
        # These functions mask a certain segment (frequency or time) in each sample of the batch.
        '''
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)
        
        '''

        # Mixup augmentation
        # Mixes two random samples in the batch with a random mixing ratio
        # Not only changes the spectrogram, but also turn the 1Hot label vector into probabilities
        #self.mixup = Mixup(cfg.mixup_prob)

    def forward(self, x):
        # (bs, channel, time)
        # Add channel dimension for CNN input
        #bs = x.shape[0]
        #x = self.impulse(x.view(bs, 1, -1)).view(bs, -1)
        x = x[:, None, :]

        # (bs, channel, mel, time)
        x = self.wav2timefreq(x)

        #if self.cfg.minmax_norm:
        #    x = min_max_norm(x, min_val=self.cfg.min, max_val=self.cfg.max)

        #if self.training:
            # Mixup augmentation
            #if self.cfg.mixup:
                # Mixup returns adapted spectrogram and label probabilities
                #x, y = self.mixup(x, y)
            # Spectrogram augmentation, e.g. MaskFrequency/MaskTime
        #    if self.cfg.specaug:
        #        x = self.freqm(x)
        #        x = self.timem(x)

        # Forward pass through the CNN backbone
        logits = self.backbone(x)

        if self.training:
            # During training has to also return the label in case they got modified during mixup
            return logits #, y
        else:
            return logits






class UltrasonicrMobileNetV2(nn.Module): #(self, mobilenetv2_model, num_classes, sample_rate=96000, n_fft=1024, hop_size=328, mel_bins=64, fmin=16000, fmax=48000):
    def __init__(self, mobilenetv2_model, num_classes, sample_rate=32000, n_fft=1024, hop_size=328,
                 mel_bins=64, fmin=50, fmax=16000):
        super().__init__()

        # Use only the CNN feature extractor from the pretrained MobileNetV2
        self.feature_extractor = mobilenetv2_model.features
        self.fc = mobilenetv2_model.fc_audioset
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.embedding_dim = mobilenetv2_model.last_channel  # e.g., 1280
       
        self.bn = nn.BatchNorm1d(num_classes)  # Optional
        self.fc1 = nn.Linear(1280, 1024)
        self.dropout = nn.Dropout(p=0.5)

        self.input_adapter = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),  # same input/output shape
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)

)
        # Log-mel frontend over 16–48kHz
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_size,
            win_length=n_fft,
            f_min=fmin,
            f_max=fmax,
            n_mels=mel_bins,
            power=2.0,
            norm='slaney',
            mel_scale='htk'
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, waveform):
       
        # Compute log-mel spectrogram [B, 1, mel_bins, time]
        mel_spec = self.mel_spectrogram(waveform)           # [B, mel_bins, T]
        logmel = self.amplitude_to_db(mel_spec)             # [B, mel_bins, T]
        x = logmel.unsqueeze(1)                             # [B, 1, mel_bins, T]

        # Optional: 1×1 conv adapter
        #x = self.input_adapter(x)                           # [B, 1, mel_bins, T]

        # CNN feature extractor
        x = self.feature_extractor(x)                       # [B, C, f', t']
        x = self.pooling(x)                                 # [B, C, 1, 1]
        x = x.squeeze(-1).squeeze(-1)                       # [B, C]

        # Classification head
        x = F.relu(self.fc1(x))                             # [B, 1024]
        x = self.dropout(x)
        logits = self.fc(x)                                 # [B, num_classes]
        logits = self.bn(logits)

        return logits

class UltrasonicContextEncoderMobileNetV2(nn.Module):
    def __init__(self, mobilenetv2_model, num_classes, stack_depth=4, slice_height=16, anchor_count=8):
        super().__init__()

        # Use only the CNN feature extractor from the pretrained MobileNetV2
        self.feature_extractor = mobilenetv2_model.features
        self.fc = mobilenetv2_model.fc_audioset
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.stack_depth = stack_depth
        self.slice_height = slice_height
        self.anchor_count = anchor_count
        self.embedding_dim = mobilenetv2_model.last_channel  # e.g., 1280
       
        self.bn = nn.BatchNorm1d(num_classes)  # Optional
        self.fc1 = nn.Linear(1280, 1024)
        self.dropout = nn.Dropout(p=0.5)

        self.input_adapter = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),  # same input/output shape
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
)


    def forward(self, ultra_stack):
        """
        Args:
            ultra_stack: [B, N_anchors, stack_depth, slice_height, T]

        Returns:
            [B, embedding_dim] - pooled representation per sample
        """
        B, N, D, H, T = ultra_stack.shape
        x = ultra_stack.view(B * N, 1, D * H, T)  # Flatten to [B*N, 1, F, T]

        x = self.input_adapter(x)    

        x = self.feature_extractor(x)  # [B*N, C, f', t']
        x = self.pooling(x)            # [B*N, C, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [B*N, C]

        x = x.view(B, N, -1)           # [B, N, C]
        x = x.mean(dim=1)             # Mean-pool over anchors -> [B, C]

        x = F.relu(self.fc1(x))                # → [B, 1024]
        x = self.dropout(x) 
        logits = self.fc(x)                # → [B, num_classes]
        logits = self.bn(logits)               # optional

        return logits  # Final ultrasonic embedding per sample




class EffNet(nn.Module):
    def __init__(self, NumClasses=550, b=0, pretrain=True, ExternalEmbeddingSize=320):
        super(EffNet, self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        if pretrain == False:
            print('EfficientNet Model Trained from Scratch (ImageNet Pretraining NOT Used).')
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=1)
            self.effnet._fc = nn.Identity()
            
        else:
            print('Now Use ImageNet Pretrained EfficientNet-B{:d} Model.'.format(b))
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=1)
            self.effnet._fc = nn.Identity()
            self.effnet._conv_head = nn.Identity()
            self.effnet._bn1 = nn.Identity()

        #self.conv = nn.Conv2d(in_channels=self.middim[b], out_channels=ExternalEmbeddingSize, kernel_size=1)
        self.fc1 = nn.Linear(2*ExternalEmbeddingSize, NumClasses)
        #self.fc1 = nn.Linear(self.middim[b] + ExternalEmbeddingSize, NumClasses)
        #self.fc1 = nn.Linear(self.middim[b], NumClasses)
        self.dropout = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm1d(NumClasses)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, emb, epoch, iteration, mode):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        #x = x.unsqueeze(1)

        if iteration == 0 and mode == 'val' and False:

            y = self.effnet.extract_features(x)
            output_tensor = y[0, :]
            output_tensor = (output_tensor - output_tensor.min()) / (output_tensor.max() - output_tensor.min())
            output_tensor = F.interpolate(output_tensor.unsqueeze(0), size=(80, 513), mode='bilinear', align_corners=False).squeeze(0)

            output_array = output_tensor.detach().cpu().numpy().transpose(1, 2, 0)
            for j in range(20):

                # Create a grid of images from the feature map array
                fig, axs = plt.subplots(8, 2, figsize=(60, 50))
                for i in range(16):
                    ax = axs[i // 2, i % 2]
                    ax.imshow(output_array[:, :, i+j], cmap='gray')
                    ax.axis('off')
                plt.subplots_adjust(wspace=0, hspace=1)
                plt.show()
                plt.savefig('feature_maps/epoch_{}_feature_map_{}.png'.format(epoch, j))


        x = self.effnet(x)
        #x = self.conv(x)
        x = self.dropout(x)
        # Concatenate the output of the last covolutional layer and the pre-computed external embedding
        if emb.dtype != x.dtype:
            emb = emb.to(x.dtype)
        x = torch.cat((x, emb.squeeze(1)), 1)
        x = normalize(x)
        x = self.fc1(x)
        x = self.bn1(x)

        x = self.softmax(x)

        return x
    
class EffNetTrained(nn.Module):
    def __init__(self, NumClasses=550, b=0, pretrain=True, ExternalEmbeddingSize=320):
        super(EffNetTrained, self).__init__()

        model = EffNet(NumClasses, b=0)
        model.load_state_dict(torch.load('/home/ubuntu/burooj/embeddings-baseline/best_model_effnet_maxpooled.pt'))
        self.effnet = model.effnet

        # Disable gradient computation for effnet
        for param in self.effnet.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(ExternalEmbeddingSize, NumClasses)

        self.dropout = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm1d(NumClasses)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, emb, epoch, iteration, mode):

        x = self.effnet(x)
        x = self.dropout(x)
        x = normalize(x)
       
        x = self.fc1(x)
        x = self.bn1(x)

        x = self.softmax(x)

        return x



class EffNetSansEmb(nn.Module):
    def __init__(self, NumClasses=550, b=0, pretrain=True, ExternalEmbeddingSize=320):
        super(EffNetSansEmb, self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        if pretrain == False:
            print('EfficientNet Model Trained from Scratch (ImageNet Pretraining NOT Used).')
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=1)
            self.effnet._fc = nn.Identity()
            
        else:
            print('Now Use ImageNet Pretrained EfficientNet-B{:d} Model.'.format(b))
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=1)
            self.effnet._fc = nn.Identity()
            self.effnet._conv_head = nn.Identity()
            self.effnet._bn1 = nn.Identity()

        #self.conv = nn.Conv2d(in_channels=self.middim[b], out_channels=ExternalEmbeddingSize, kernel_size=1)
        self.fc1 = nn.Linear(ExternalEmbeddingSize, NumClasses)
        #self.fc1 = nn.Linear(self.middim[b] + ExternalEmbeddingSize, NumClasses)
        #self.fc1 = nn.Linear(self.middim[b], NumClasses)
        self.dropout = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm1d(NumClasses)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, emb, epoch, iteration, mode):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        #x = x.unsqueeze(1)
        x = self.effnet(x)
        #x = self.conv(x)
        x = self.dropout(x)
        # Concatenate the output of the last covolutional layer and the pre-computed external embedding
        # if emb.dtype != x.dtype:
        #     emb = emb.to(x.dtype)
        # x = torch.cat((x, emb.squeeze(1)), 1)
        # x = normalize(x)
        x = self.fc1(x)
        x = self.bn1(x)

        x = self.softmax(x)

        return x



class FullyConnected(nn.Module):
    def __init__(self, NumClasses=550, pretrain=True, ExternalEmbeddingSize=320):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(ExternalEmbeddingSize, NumClasses)
        #self.fc1 = nn.Linear(self.middim[b] + ExternalEmbeddingSize, NumClasses)
        #self.fc1 = nn.Linear(self.middim[b], NumClasses)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(NumClasses)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, emb):
        
        x = self.fc1(emb.squeeze(1))
        x = self.bn1(x)

        #x = self.softmax(x)

        return x    



class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV3Small, self).__init__()
        
        self.mobilenet = models.mobilenet_v3_small(pretrained=True)

        # Modify the first convolutional layer to accept single-channel inputs
        self.mobilenet.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)

        # Replace the last layer with a new linear layer
        in_features = self.mobilenet.classifier[3].in_features
        self.mobilenet.classifier[3] = nn.Linear(in_features, num_classes)
        self.batchnorm = nn.BatchNorm1d(num_classes)

        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x, emb, epoch, iteration, mode):
        x = self.mobilenet(x)
        x = self.batchnorm(x)
        x = self.softmax(x)
        
        return x
    
class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        
        self.densenet = models.densenet121(pretrained=True)

        # Modify the first convolutional layer to accept single-channel inputs
        self.densenet.features[0] = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)

        # Replace the last layer with a new linear layer
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)
        self.batchnorm = nn.BatchNorm1d(num_classes)
        self.softmax = nn.Softmax(dim=1)
       
    def forward(self, x, emb, epoch, iteration, mode):

        x = self.densenet(x)
        x = self.batchnorm(x)
        x = self.softmax(x)
        
        return x   

class DenseNet121Embedding(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121Embedding, self).__init__()
        
        self.densenet = models.densenet121(pretrained=True)

        model = DenseNet121(num_classes)  
        model.load_state_dict(torch.load('/home/ubuntu/burooj/embeddings-baseline/best_model_densenet_maxpooled_Aug.pt'))
        self.densenet = model.densenet
        # Modify the first convolutional layer to accept single-channel inputs
        #self.densenet.features[0] = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)

        # Disable gradient computation for effnet
        for param in self.densenet.parameters():
            param.requires_grad = False
        # Replace the last layer with a new linear layer
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()
        self.fc = nn.Linear(in_features + 320, num_classes)
        self.batchnorm = nn.BatchNorm1d(num_classes)
        self.softmax = nn.Softmax(dim=1)
       
    def forward(self, x, emb, epoch, iteration, mode):

        x = self.densenet(x)
        # Concatenate the output of the last covolutional layer and the pre-computed external embedding
        if emb.dtype != x.dtype:
            emb = emb.to(x.dtype)
        x = torch.cat((x, emb.squeeze(1)), 1)
        x = normalize(x)
        x = self.fc(x)
        x = self.batchnorm(x)
        x = self.softmax(x)
        
        return x         

class DualBranchDenseNet(nn.Module):
    def __init__(self, NumClasses=438, b=0, pretrain=True, ExternalEmbeddingSize=320):
        super(DualBranchDenseNet, self).__init__()
       
        
        self.densenet = models.densenet121(pretrained=True)

        # Modify the first convolutional layer to accept single-channel inputs
        self.densenet.features[0] = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)

        # Replace the last layer with a new linear layer
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier =  nn.Identity()

        self.classifier = nn.Linear(in_features + ExternalEmbeddingSize, NumClasses)
        self.batchnorm = nn.BatchNorm1d(NumClasses)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x, emb, epoch, iteration, mode):
        x = self.densenet(x)

        # Concatenate the output of the last covolutional layer and the pre-computed external embedding
        if emb.dtype != x.dtype:
            emb = emb.to(x.dtype)
        x = torch.cat((x, emb.squeeze(1)), 1)
        x = normalize(x)
        x = self.classifier(x)
        x = self.batchnorm(x)
        x = self.softmax(x)

        return x

class DualBranchDenseNet2(nn.Module):
    def __init__(self, NumClasses=438, b=0, pretrain=True, ExternalEmbeddingSize=320):
        super(DualBranchDenseNet2, self).__init__()
       
        
        self.densenet = models.densenet121(pretrained=True)

        # Modify the first convolutional layer to accept single-channel inputs
        self.densenet.features[0] = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)

        # Replace the last layer with a new linear layer
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier =  nn.Identity()

        self.classifier1 = nn.Linear(ExternalEmbeddingSize, NumClasses)
        self.classifier2 = nn.Linear(in_features, NumClasses)
        self.classifier_out = nn.Linear(NumClasses, NumClasses)    
        self.batchnorm = nn.BatchNorm1d(NumClasses)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x, emb, epoch, iteration, mode):
        x = self.densenet(x)
        x = self.classifier2(x)

        # Concatenate the output of the last covolutional layer and the pre-computed external embedding
        if emb.dtype != x.dtype:
            emb = emb.to(x.dtype)

        y = self.classifier1(emb.squeeze(1))
        x = x + y
        #x = normalize(x)
        x = self.classifier_out(x)
        x = self.batchnorm(x)
        x = self.softmax(x)

        return x        




class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained 
    differently during ImageNet pretraining.
    """
    
    def __init__(self, label_dim=438, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, 
                 model_size='base384', verbose=True, shift_prob=0.5, c_noise_prob=0.5, b_noise_prob=0.5, augmentation=False):

        super(ASTModel, self).__init__()
#         assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'
        self.melspectrogramer = torchaudio.transforms.MelSpectrogram(sample_rate=32000, n_fft=2048, win_length=2048, 
                                                                     hop_length=512, f_min=FMIN, 
                                                                     f_max=FMAX, n_mels=128, power=1.0)
       
        self.amplituder = torchaudio.transforms.AmplitudeToDB()
        self.augmentation = augmentation
        self.freqm = 40
        self.timem = 100
        self.noise = AddColoredNoise(p=c_noise_prob, p_mode="per_example", mode="per_example", sample_rate=SAMPLE_RATE,
                                     min_snr_in_db=25, max_snr_in_db=40, min_f_decay=-2, max_f_decay=1.5)
        self.shift = Shift(p=shift_prob, p_mode="per_example", mode="per_example", sample_rate=SAMPLE_RATE)
        self.background_noise = AddBackgroundNoise(background_paths="/data/burooj/ebbc2-xenocanto-chunks-549/noise", min_snr_in_db=3.0, max_snr_in_db=30.0, p=b_noise_prob, sample_rate=SAMPLE_RATE)
        self.spec_aug = torch.nn.Sequential(
            torchaudio.transforms.TimeStretch(0.8, fixed_rate=True),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=80),
            torchaudio.transforms.TimeMasking(time_mask_param=80),
        )
        
        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, 
                                                                            self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, 
                                                                                                                                 self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists('./audioset_10_10_0.4593.pth') == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='./audioset_10_10_0.4593.pth')
            sd = torch.load('./audioset_10_10_0.4593.pth', map_location=device)
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                                   imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            self.bn = nn.BatchNorm1d(label_dim)

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x, is_training=False):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        """
        if is_training and self.augmentation:
            x = self.noise(x.unsqueeze(0)).squeeze(0)
            x = self.shift(x.unsqueeze(0)).squeeze(0)
            x = self.background_noise(x.unsqueeze(0)).squeeze(0)

        x = self.melspectrogramer(x)
        #x = torchaudio.compliance.kaldi.fbank(x, htk_compat=True, sample_frequency=32000, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=8, frame_length=16)
        
        # Time and frequency masking + time stretch only for training set
        #if is_training and self.augmentation:
        #    x = self.spec_aug(x)

        
        # Time and frequency masking, only for training set
        if is_training and self.augmentation:
            # SpecAug, not do for eval set
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            #spec = torch.transpose(spec, 0, 1)
            if self.freqm != 0:
                x = freqm(x)
            if self.timem != 0:
                x = timem(x)
        """        
        x = self.amplituder(x)
#         plt.imshow(x[0].detach().cpu())
#         plt.show()

        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        #x = x.unsqueeze(1) # uncomment this if you go back to computing melspec here instead of computing in the dataloader
        
        #x_mean = x.mean(keepdim=True, dim=(1, 2, 3))
        #x_std = x.std(keepdim=True, dim=(1, 2, 3))
        #x = (x - x_mean) / (x_std * 2 + 1e-6)
        
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        #x = self.bn(x)

        return x


