"""
noise_config.py

Configuration file for the noise added during the game.
"""
from configs.base_config import BaseConfig


class NoiseConfig(BaseConfig):
    """Noise-specific configuration parameters."""
    
    __slots__ = {
        'angle', 'distance', 'proximity', 'time'
    }
    
    def __init__(self):
        # Alpha in Gaussian distribution for angular sensor, max 0.7° deviation  [def=0.001]
        self.angle: float = 0.001
        # Alpha in Gaussian distribution for distance sensor, max ~1.5cm deviation  [def=0.005]
        self.distance: float = 0.005
        # Alpha in Gaussian distribution for proximity sensor, max ~1.5cm deviation  [def=0.005]
        self.proximity: float = 0.005
        # Alpha in Gaussian distribution for time noise, max 0.02s deviation  [def=0.005]
        self.time: float = 0.005
