"""
E-Scooter Safety Detection System
Source package initialization
"""

__version__ = "1.0.0"
__author__ = "BEAM Detection System Team"
__license__ = "MIT"

from .detector import EScooterDetector
from .violation_checker import ViolationChecker
from .alert_system import AlertSystem
from .model_loader import ModelLoader
from .video_handler import VideoHandler

__all__ = [
    'EScooterDetector',
    'ViolationChecker',
    'AlertSystem',
    'ModelLoader',
    'VideoHandler'
]
