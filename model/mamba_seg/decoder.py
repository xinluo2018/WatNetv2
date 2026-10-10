"""Small additive feature pyramid for segmentation experiments."""
from torch import nn
from torch.nn import functional as F


class PyramidDecoder(nn.Module):
    def __init__(self, in_channels, channels, num_classes):
        super().__init__()
        if not in_channels or channels < 1 or num_classes < 1:
            raise ValueError('Require feature channels and positive decoder/classes widths')
        self.projections = nn.ModuleList(nn.Conv2d(c, channels, 1) for c in in_channels)
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, num_classes, 1))

    def forward(self, features):
        if len(features) != len(self.projections):
            raise ValueError('Feature count does not match decoder projections')
        x = self.projections[-1](features[-1])
        for projection, feature in reversed(list(zip(self.projections[:-1], features[:-1]))):
            x = projection(feature) + F.interpolate(
                x, size=feature.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(x)
