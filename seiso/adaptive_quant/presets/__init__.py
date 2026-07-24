"""Curated :class:`FrameworkConfig` presets for research CLI entrypoints and examples.

Import named constants from this package
(``from seiso.adaptive_quant.presets import CONFIG``).

**Research-only:** ``CONFIG_3090``, ``CONFIG_4090``, ``CONFIG_4090_UNIVERSAL``,
``CONFIG_GPU``, ``CONFIG_MOE``, ``CONFIG_ONLINE`` (and related helpers) are for
``python -m seiso.adaptive_quant`` / research CLIs. They are **not** Forge /
``seiso rl-quant`` product preset IDs — those live in ``seiso.rl_quant.presets``
(``reproducible`` / ``minimal`` / ``post_train``) (RP-09).
"""

from seiso.adaptive_quant.presets.baseline import CONFIG
from seiso.adaptive_quant.presets.continuous import CONFIG_CONTINUOUS
from seiso.adaptive_quant.presets.gpu import CONFIG_GPU, make_rtx_torch_preset
from seiso.adaptive_quant.presets.moe import CONFIG_MOE
from seiso.adaptive_quant.presets.online import CONFIG_ONLINE
from seiso.adaptive_quant.presets.post_train import CONFIG_POST_TRAIN
from seiso.adaptive_quant.presets.rtx3090 import CONFIG_3090
from seiso.adaptive_quant.presets.rtx4090 import CONFIG_4090
from seiso.adaptive_quant.presets.rtx4090_universal import CONFIG_4090_UNIVERSAL

__all__ = [
    "CONFIG",
    "CONFIG_3090",
    "CONFIG_4090",
    "CONFIG_4090_UNIVERSAL",
    "CONFIG_CONTINUOUS",
    "CONFIG_GPU",
    "CONFIG_MOE",
    "CONFIG_ONLINE",
    "CONFIG_POST_TRAIN",
    "make_rtx_torch_preset",
]
