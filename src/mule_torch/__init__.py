"""mule-torch: a faithful PyTorch port of MULE (SF-NFNet-F0).

Public API::

    from mule_torch import MuleModel, SfNfNetF0, MuleConfig
    model = MuleModel.from_pretrained(model_dir="...")
    emb = model(waveform)   # (B, T) @ 16 kHz -> (B, 1728)
"""

from __future__ import annotations

from mule_torch.backbone import SfNfNetF0
from mule_torch.config import MuleConfig
from mule_torch.model import MuleModel

__version__ = "0.2.0"
__all__ = ["MuleModel", "SfNfNetF0", "MuleConfig", "__version__"]
