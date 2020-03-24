"""
bot_config.py

MarXBot configuration file.
"""
from configs.base_config import BaseConfig


class BotConfig(BaseConfig):
    """Robot-specific configuration parameters."""
    
    __slots__ = {
        'driving_speed', 'radius', 'ray_distance', 'turning_speed',
    }
    
    def __init__(self):
        # Maximal driving speed (driving straight) of the robot expressed in m/s  [def=0.6]
        self.driving_speed: float = 0.6
        # Radius of the bot expressed in meters  [def=0.085]
        self.radius: float = 0.085
        # Distance a ray-sensor reaches, expressed in meters  [def=1.0]
        self.ray_distance: float = 1.0
        # Maximal turning speed of the robot expressed in radians per second  [def=3.53~=0.6/0.17]
        self.turning_speed: float = 3.53
