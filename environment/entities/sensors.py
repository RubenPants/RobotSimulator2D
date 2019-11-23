"""
sensors.py

Sensor classes used by the bots. The different types of sensors are:
 * Angular: Measures the burden angle between the robot and the target
 * Distance: Measures the distance in crows flight between the robot and the target
 * Proximity: Measures the proximity of walls in a certain direction of the robot
"""
import random

import numpy as np

from utils.config import NOISE_SENSOR_ANGLE, NOISE_SENSOR_DIST, NOISE_SENSOR_PROXY, SENSOR_RAY_DISTANCE
from utils.dictionary import *
from utils.intersection import line_line_intersection
from utils.line2d import Line2d
from utils.vec2d import Vec2d


class Sensor:
    """
    The baseclass used by all sensors.
    """
    
    def __init__(self,
                 game,  # Type not specified due to circular imports
                 sensor_id: int = 0,
                 angle: float = 0,
                 pos_offset: float = 0,
                 max_dist: float = 0):
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
    
    def __str__(self):
        """
        :return: Name of the sensor
        """
        raise NotImplemented
    
    def get_measure(self):
        """
        Read the distance to the first object in the given space. Give visualization of the sensor if VISUALIZE_SENSOR
        is set on True.
        
        :return: Distance
        """
        raise NotImplemented


class AngularSensor(Sensor):
    """
    Angle deviation between bot and wanted direction in 'crow flight'.
    """
    
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
        super().__init__(game=game,
                         sensor_id=sensor_id)
        self.clockwise = clockwise
    
    def __str__(self):
        return "{sensor}_{id:02d}".format(sensor=D_SENSOR_ANGLE, id=self.id)
    
    def get_measure(self):
        """
        :return: Float between 0 and 2*PI
        """
        # Get relative angle
        start_a = self.game.player.angle
        req_a = (self.game.target - self.game.player.pos).get_angle()
        
        # Normalize
        diff = 2 * np.pi + start_a - req_a
        diff %= 2 * np.pi
        
        # Check direction
        if not self.clockwise:
            diff = abs(2 * np.pi - diff)
            diff %= 2 * np.pi
        
        # Add noise
        if self.game.noise:
            diff += random.gauss(0, NOISE_SENSOR_ANGLE)
        return diff


class DistanceSensor(Sensor):
    """
    Distance from bot to the target in 'crows flight'.
    """
    
    def __init__(self,
                 game,  # Type not specified due to circular imports
                 sensor_id: int = 0):
        """
        :param game: Reference to the game in which the sensor is used
        :param sensor_id: Identification number for the sensor
        """
        super().__init__(game=game,
                         sensor_id=sensor_id)
    
    def __str__(self):
        return "{sensor}_{id:02d}".format(sensor=D_SENSOR_DISTANCE, id=self.id)
    
    def get_measure(self):
        """
        :return: Distance between target and robot's center coordinate, which is a float
        """
        start_p = self.game.player.pos
        end_p = self.game.target
        distance = (start_p - end_p).get_length()
        if self.game.noise:
            distance += random.gauss(0, NOISE_SENSOR_DIST)
        return distance


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
                 max_dist: float = SENSOR_RAY_DISTANCE):
        """
        :param game: Reference to the game in which the sensor is used
        :param sensor_id: Identification number for the sensor
        :param angle: Relative angle to the agent's center of mass and orientation
        :param pos_offset: Distance to the agent's center of mass and orientation
        :param max_dist: Maximum distance the sensor can reach, infinite if set to zero
        """
        super().__init__(game=game,
                         sensor_id=sensor_id,
                         angle=angle,
                         pos_offset=pos_offset,
                         max_dist=max_dist)
        self.start_pos = None  # Placeholder for start-point of proximity sensor
        self.end_pos = None  # Placeholder for end-point of proximity sensor
    
    def __str__(self):
        return "{sensor}_{id:02d}".format(sensor=D_SENSOR_PROXIMITY, id=self.id)
    
    def get_measure(self):
        """
        Get the distance to the closest wall. If all the walls are 'far enough', as determined by self.max_dist, then
        the maximum sensor-distance is returned.
        
        :return: Float expressing the distance to the closest wall, if there is any
        """
        # Start and end point of ray
        normalized_offset = Vec2d(np.cos(self.game.player.angle + self.angle),
                                  np.sin(self.game.player.angle + self.angle))
        self.start_pos = self.game.player.pos + normalized_offset * self.pos_offset
        self.end_pos = self.game.player.pos + normalized_offset * (self.pos_offset + self.max_dist)
        sensor_line = Line2d(x=self.game.player.pos,
                             y=self.end_pos)
        
        # Check if there is a wall intersecting with the sensor and return the closest distance to a wall
        closest_dist = self.max_dist
        for wall in self.game.walls:
            inter, pos = line_line_intersection(sensor_line, wall)
            if inter:
                new_dist = (pos - self.start_pos).get_length()
                if closest_dist > new_dist:
                    self.end_pos = pos
                    closest_dist = new_dist
        
        if self.game.noise:
            closest_dist += random.gauss(0, NOISE_SENSOR_PROXY)
        return closest_dist
