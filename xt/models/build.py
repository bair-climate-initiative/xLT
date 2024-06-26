from dataclasses import dataclass, field

from . import backbones
from .context_encoders import ContextEncoderConfig
from .decoders.decoder import xT


@dataclass
class BackboneConfig:
    """Configuration for feature extracting backbone."""

    in_chans: int = 3
    """Number of channels in input data."""
    input_dim: int = 2
    """Input dimension."""
    drop_path_rate: float = 0.0
    """Drop path rate for stochastic depth."""
    pretrained: str = ""
    """Path to pretrained weights, empty for none."""
    channel_last: bool = True
    """If channels are last in data format."""
    input_size: int = 256
    """Expected input size of data."""

    self_supervised: bool = False
    """Whether or not the backbone should be trained using a self-supervised loss."""
    ssl_bottleneck: int = 512
    """The size of the Linear layer used to predict the attention layer of the context encoder."""


@dataclass
class ModelConfig:
    name: str = "xT"
    """Name of overarching model architecture."""
    resume: str = ""
    """Path to checkpoint to resume training from. Empty for none."""
    tiling: str = "naive"
    """Transformer-XL tiling strategy"""
    backbone_class: str = "swinv2_tiny_window16_256_timm"
    """Class name for backbone."""
    patch_size: int = 16
    """Patch size used for transformer XL."""  # TODO: properly derive this
    num_classes: int = 9999
    cls_head: str = "naive"
    """Number of classes for head on dataset."""
    mlp_ratio: int = 4
    """MLP ratio for Enc/Dec."""

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    context: ContextEncoderConfig = field(default_factory=ContextEncoderConfig)


def build_model(config: ModelConfig, dataset: str = "inaturalist"):
    backbone_class = config.backbone_class
    backbone = backbones.__dict__[backbone_class](**config.backbone, hidden_size=config.context.hidden_size)

    if config.name == "xT":
        model = xT(
            backbone=backbone,
            xl_config=config.context,
            channels_last=config.backbone.channel_last,
            crop_size=config.backbone.input_size,
            skip_decoder=False,
            backbone_name=config.backbone_class,
            dataset=dataset,
            num_classes=config.num_classes,
            mlp_ratio=config.mlp_ratio,
            cls_head=config.cls_head,
            self_supervised=config.backbone.self_supervised,
            ssl_bottleneck=config.backbone.ssl_bottleneck,
        )
    return model
