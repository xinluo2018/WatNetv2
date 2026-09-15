"""Framework-independent segmentation entry point. Inputs/outputs use NCHW."""
from torch import nn
from torch.nn import functional as F

from .models.vmamba.backbone import VMamba
from .models.decoder import PyramidDecoder


class MambaSeg(nn.Module):
    """Return full-resolution logits; either branch can be replaced.
    Custom backbones expose ``out_channels`` and return a sequence of NCHW maps.
    Custom decoders accept that sequence and return NCHW class logits.
    """
    def __init__(self, num_classes=2, in_chans=3, decoder_channels=128,
                 backbone=None, decoder=None, **backbone_kwargs):
        super().__init__()
        if num_classes < 1:
            raise ValueError('num_classes must be positive')
        if backbone is not None and backbone_kwargs:
            raise ValueError('Configure a custom backbone before passing it in')
        self.backbone = backbone if backbone is not None else VMamba(
            in_chans=in_chans, **backbone_kwargs)
        self.decoder = decoder if decoder is not None else PyramidDecoder(
            self.backbone.out_channels, decoder_channels, num_classes)
        if backbone is None:
            self.backbone.init_weights()

    def forward(self, x):
        logits = self.decoder(self.backbone(x))
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
