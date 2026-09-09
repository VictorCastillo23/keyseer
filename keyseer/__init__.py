"""
keyseer
=======

Deteccion de keyframes por consolidacion de componentes en modelos de
mezcla gaussiana online. Ver README.md para la idea y el uso.
"""

from .keyframes import KeyframeResult, extract_keyframes
from .core import KeySeer, KeySeerConfig, FrameReadout

__all__ = [
    "extract_keyframes",
    "KeyframeResult",
    "KeySeer",
    "KeySeerConfig",
    "FrameReadout",
]
