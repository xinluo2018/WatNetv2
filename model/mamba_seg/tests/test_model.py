"""CPU checks: python -m unittest discover -s model/mamba_seg/tests -v"""
import unittest
import sys
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mamba_seg import MambaSeg, VMamba
from mamba_seg.models.vmamba.vss_block import DropPath
from mamba_seg.models.decoder import UPerNetDecoder


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.options = dict(dims=8, depths=(1, 1, 1, 1), backend='torch', drop_path_rate=0.)

    def test_multispectral_logits_and_gradients(self):
        model = MambaSeg(in_chans=6, num_classes=3, decoder_channels=8, **self.options)
        x = torch.randn(2, 6, 17, 23, requires_grad=True)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 3, 17, 23))
        target = torch.randint(3, (2, 17, 23))
        nn.functional.cross_entropy(logits, target).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_custom_mixer_and_feature_subset(self):
        class LocalMixer(nn.Module):
            def __init__(self, dim, *args, **kwargs):
                super().__init__()
                self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

            def forward(self, x):
                return self.conv(x)

        backbone = VMamba(out_indices=(0, 2), block_kwargs=dict(mixer_cls=LocalMixer), **self.options)
        self.assertEqual(backbone.out_channels, (8, 32))
        self.assertEqual(backbone.feature_strides, (4, 16))
        model = MambaSeg(backbone=backbone, num_classes=1, decoder_channels=8)
        self.assertEqual(model(torch.randn(1, 3, 19, 27)).shape, (1, 1, 19, 27))

    def test_custom_decoder(self):
        class Head(nn.Module):
            def forward(self, features):
                return features[0][:, :2]

        model = MambaSeg(decoder=Head(), **self.options)
        self.assertEqual(model(torch.randn(1, 3, 16, 20)).shape, (1, 2, 16, 20))

    def test_upernet_decoder(self):
        decoder = UPerNetDecoder((8, 16, 32, 64), channels=8, num_classes=3)
        decoder.eval()
        features = [
            torch.randn(2, 8, 16, 20), torch.randn(2, 16, 8, 10),
            torch.randn(2, 32, 4, 5), torch.randn(2, 64, 2, 3),
        ]
        self.assertEqual(decoder(features).shape, (2, 3, 16, 20))

    def test_checkpoint_backward(self):
        a = MambaSeg(decoder_channels=8, **self.options)
        b = MambaSeg(decoder_channels=8, use_checkpoint=True, **self.options)
        b.load_state_dict(a.state_dict())
        x = torch.randn(1, 3, 17, 19)
        y, z = a(x), b(x)
        torch.testing.assert_close(y, z)
        y.square().mean().backward()
        z.square().mean().backward()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad)

    def test_drop_path(self):
        layer = DropPath(.5)
        x = torch.ones(64, 3, 2, 2)
        y = layer(x)
        self.assertTrue(((y == 0) | (y == 2)).all())
        torch.testing.assert_close(y, y[:, :1, :1, :1].expand_as(y))
        torch.testing.assert_close(layer.eval()(x), x)


if __name__ == '__main__':
    unittest.main()
