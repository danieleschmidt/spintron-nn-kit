"""
SpinTron-NN-Kit: Ultra-low-power neural inference framework for spin-orbit-torque hardware.

This package provides an end-to-end flow from PyTorch models to spintronic hardware
implementations using magnetic tunnel junction (MTJ) crossbars.
"""

from .core.mtj_models import MTJConfig, MTJDevice
from .core.crossbar import MTJCrossbar
from .converter.pytorch_parser import SpintronConverter
from .models.vision import MobileNetV2_Spintronic
from .training.qat import SpintronicTrainer

__version__ = "0.1.0"
__author__ = "SpinTron-NN-Kit Contributors"
__email__ = "spintron-nn@example.com"

__all__ = [
    "MTJConfig",
    "MTJDevice", 
    "MTJCrossbar",
    "SpintronConverter",
    "MobileNetV2_Spintronic",
    "SpintronicTrainer",
]