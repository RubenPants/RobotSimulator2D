"""
sensors.py

Sensor classes used by the bots. The different types of sensors are:
 * Angular: Measures the burden angle between the robot and the target
 * Distance: Measures the distance in crows flight between the robot and the target
 * Proximity: Measures the proximity of walls in a certain direction of the robot
"""
import random

from numpy import cos, pi, sin

from utils.intersection import line_line_intersection
from utils.line2d import Line2d
from utils.vec2d import Vec2d


class Sensor:
    """The baseclass used by all sensors."""
    
    __slots__ = (
        "game",
        "id", "angle", "pos_offset", "max_dist", "value",
    )
    
    def __init__(self,
                 game,  # Type not specified due to circular imports
                 sensor_id: int = 0,
                 angle: float = 0,
                 pos_offset: float = 0,
                 max_dist: float = 0
                 ):
        """
        Basic characteristics of a sensor.
        
        :param game: Reference to the game in which the sensor is used
        :param sensor_id: Identification number for the sensor
        :param angle: Relative angle to the agent's center of mass
        :param pos_offset: Distance to the agent's center of mass
        :param max_dist: Maximum distance the sensor can reach, infinite if set to zero
        """
        # Game-object
        self.game = game
        
        # Default sensor attributes
        self.id = sensor_id
        self.angle = angle
        self.pos_offset = pos_offset
        self.max_dist = max_dist
        self.value = 0.0  # Zero value for initialized sensors
    
    def __str__(self):
        """ :return: Name of the sensor """
        raise NotImplemented
    
    def measure(self, close_walls: set = None):
        """Store the sensor's current value in self.value. If the surrounding walls are known these can be given."""
        raise NotImplemented


class AngularSensor(Sensor):
    """Angle deviation between bot and wanted direction in 'crows flight'."""
    
    def __init__(self,
                 game,  # Type not specified due to circular imports
                 sensor_id: int = 0,
                 clockwise: bool = True):
        """
        :param clockwise: Calculate the angular difference in clockwise direction
        :param game: Reference to the game in which the sensor is used
        :param sensor_id: Identification number for the sensor
        """
        # noinspection PyCompatibility
        super().__init__(game=game, sensor_id=sensor_id)
        self.clockwise = clockwise
    
    def __str__(self):
        return f"angular {'right' if self.clockwise else 'left'}"
    
    def measure(self, close_walls: set = None):
        """Update self.value, result is a float between 0 and 2*PI."""
        # Get relative angle
        start_a = self.game.player.angle
        req_a = (self.game.target - self.game.player.pos).get_angle()
        
        # Normalize
        self.value = 2 * pi + start_a - req_a
        self.value %= 2 * pi
        
        # Check direction
        if not self.clockwise:
            self.value = abs(2 * pi - self.value)
            self.value %= 2 * pi
        
        # Add noise
        if self.game.noise: self.value += random.gauss(0, self.game.noise_angle)


class DeltaDistanceSensor(Sensor):
    """Difference in distance from bot to the target in 'crows flight' between current and the previous time-point."""
    
    def __init__(self,
                 game,  # Type not specified due to circular imports
                 sensor_id: int = 0):
        """
        :param game: Reference to the game in which the sensor is used
        :param sensor_id: Identification number for the sensor
        """
        super().__init__(game=game, sensor_id=sensor_id)
        self.distance: float = 0.0
        self.prev_distance: float = 0.0
    
    def __str__(self):
        return "delta_distance"
    
    def measure(self, close_walls: set = None):
        """Update self.value to difference between previous distance and current distance."""
        self.prev_distance = self.distance  # Save previous distance
        start_p = self.game.player.pos
        end_p = self.game.target
        self.distance = (start_p - end_p).get_length()  # Get current measure
        if self.prev_distance == 0.0: self.prev_distance = self.distance  # Disable cold start
        self.value = self.prev_distance - self.distance  # Positive value == closer to target


class DistanceSensor(Sensor):
    """Distance from bot to the target in 'crows flight'."""
    
    def __init__(self,
                 game,  # Type not specified due to circular imports
                 normalizer: float,
                 sensor_id: int = 0):
        """
        :param game: Reference to the game in which the sensor is used
        :param normalizer: The constant by which the distance-value is normalized
        :param sensor_id: Identification number for the sensor
        """
        super().__init__(game=game, sensor_id=sensor_id)
        self.normalizer = normalizer
    
    def __str__(self):
        return "distance"
    
    def measure(self, close_walls: set = None):
        """Update self.value to current distance between target and robot's center coordinate."""
        start_p = self.game.player.pos
        end_p = self.game.target
        self.value = (start_p - end_p).get_length() / self.normalizer
        if self.game.noise: self.value += random.gauss(0, self.game.noise_distance)


class ProximitySensor(Sensor):
    """
    The proximity sensor is attached to a bot's edge and measures the distance min(max_distance, object_distance). In
    other words, it returns the distance to an object in its path (only a straight line) if this distance is within a
    certain threshold, otherwise the maximum value will be returned.
    """
    
    def __init__(self,
                 game,  # Type not specified due to circular imports
                 sensor_id: int = 0,
                 angle: float = 0,
                 pos_offset: float = 0,
                 max_dist: float = 0):
        """
        :param game: Reference to the game in which the sensor is used
        :param sensor_id: Identification number for the sensor
        :param angle: Relative angle to the agent's center of mass and orientation
        :param pos_offset: Distance to the agent's center of mass and orientation
        :param max_dist: Maximum distance the sensor can reach, infinite if set to zero
        """
        if not max_dist: max_dist = game.ray_distance
        super().__init__(game=game,
                         sensor_id=sensor_id,
                         angle=angle,
                         pos_offset=pos_offset,
                         max_dist=max_dist)
        self.start_pos = None  # Placeholder for start-point of proximity sensor
        self.end_pos = None  # Placeholder for end-point of proximity sensor
    
    def __str__(self):
        return f"proximity {round(self.angle * 180 / pi)}"
    
    def measure(self, close_walls: set = None):
        """
        Get the distance to the closest wall. If all the walls are 'far enough', as determined by self.max_dist, then
        the maximum sensor-distance is returned.
        
        :param close_walls: Walls which fall within ray_distance from the agent, speeds up readings
        :return: Float expressing the distance to the closest wall, if there is any
        """
        # Start and end point of ray
        normalized_offset = Vec2d(cos(self.game.player.angle + self.angle),
                                  sin(self.game.player.angle + self.angle))
        self.start_pos = self.game.player.pos + normalized_offset * self.pos_offset
        self.end_pos = self.game.player.pos + normalized_offset * (self.pos_offset + self.max_dist)
        sensor_line = Line2d(x=self.game.player.pos, y=self.end_pos)
        
        # Check if there is a wall intersecting with the sensor and return the closest distance to a wall
        self.value = self.max_dist
        for wall in close_walls if close_walls else self.game.walls:
            inter, pos = line_line_intersection(sensor_line, wall)
            if inter:
                new_dist = (pos - self.start_pos).get_length()
                if self.value > new_dist:
                    self.end_pos = pos
                    self.value = new_dist
        
        if self.game.noise:
            self.value += random.gauss(0, self.game.noise_proximity)
            self.value = max(0, min(self.value, self.max_dist))
