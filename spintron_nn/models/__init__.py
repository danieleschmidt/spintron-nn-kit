"""Pre-optimized spintronic neural network models."""

from .vision import MobileNetV2_Spintronic, TinyConvNet_Spintronic
from .audio import KeywordSpottingNet_Spintronic, WakeWordDetector_Spintronic

__all__ = [
    "MobileNetV2_Spintronic",
    "TinyConvNet_Spintronic",
    "KeywordSpottingNet_Spintronic", 
    "WakeWordDetector_Spintronic",
]