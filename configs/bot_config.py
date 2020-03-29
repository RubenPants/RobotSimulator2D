"""
bot_config.py

MarXBot configuration file.
"""
from numpy import pi

from configs.base_config import BaseConfig


class BotConfig(BaseConfig):
    """Robot-specific configuration parameters."""
    
    __slots__ = {
        'driving_speed', 'radius', 'ray_distance', 'turning_speed',
        'angular_dir', 'delta_dist_enabled', 'prox_angles',
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
        
        # Sensor-configurations
        # The clockwise directions for the angular sensors  [def=[True, False]]
        self.angular_dir = [True, False]
        # The delta-distance sensor  [def=False]
        self.delta_dist_enabled = False
        # Angles used for the proximity-sensors  [def=/]
        self.prox_angles = get_proximity_angles()


def get_proximity_angles():
    """Get the angles used for the proximity sensors."""
    angles = []
    
    # Left-side of the agent
    angles.append(3 * pi / 4)  # 135° (counter-clockwise)
    for i in range(5):  # 90° until 10° with hops of 20° (total of 5 sensors)
        angles.append(pi / 2 - i * pi / 9)
    
    # Center
    angles.append(0)  # 0°
    
    # Right-side of the agent
    for i in range(5):  # -10° until -90° with hops of 20° (total of 5 sensors)
        angles.append(-pi / 18 - i * pi / 9)
    angles.append(-3 * pi / 4)  # -135° (clockwise)
    return angles
