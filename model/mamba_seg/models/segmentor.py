# Adapted from MzeroMiko/VMamba (MIT); see LICENSE and README.md.
"""Register only the backbone. EncoderDecoder/UPerHead/FCNHead belong to MMSeg."""
from mmseg.registry import MODELS
from .vmamba.backbone import VMamba


@MODELS.register_module()
class MM_VMamba(VMamba):
    pass
