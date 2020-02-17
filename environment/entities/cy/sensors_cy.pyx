"""
sensors_cy.pyx

Cython version of the sensors.py file. Note that this file co-exists with a .pxd file (needed to import the sensor
classes and methods in other files).


"""
import random

import numpy as np

from utils.dictionary import *
from utils.cy.intersection_cy cimport line_line_intersection_cy
from utils.cy.line2d_cy cimport Line2dCy


cdef class SensorCy:
    """
    The baseclass used by all sensors.
    """
    
    def __init__(self,
                 GameCy game,  # Type not specified due to circular imports
                 int sensor_id=0,
                 float angle=0,
                 float pos_offset=0,
                 float max_dist=0):
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
        """ :return: Name of the sensor """
        raise NotImplemented
    
    cdef float get_measure(self):
        """
        Read the distance to the first object in the given space. Give visualization of the sensor if VISUALIZE_SENSOR
        is set on True.
        
        :return: Distance
        """
        raise NotImplemented


cdef class AngularSensorCy(SensorCy):
    """
    Angle deviation between bot and wanted direction in 'crow flight'.
    """
    
    def __init__(self,
                 GameCy game,  # Type not specified due to circular imports
                 int sensor_id=0,
                 bint clockwise=True):
        """
        :param clockwise: Calculate the angular difference in clockwise direction
        :param game: Reference to the game in which the sensor is used
        :param sensor_id: Identification number for the sensor
        """
        # noinspection PyCompatibility
        super().__init__(game=game, sensor_id=sensor_id)
        self.clockwise = clockwise
    
    def __str__(self):
        return f"{D_SENSOR_ANGLE}_{self.id:02d}"
    
    cdef float get_measure(self):
        """
        :return: Float between 0 and 2*PI
        """
        cdef float start_a
        cdef float req_a
        cdef float diff
        
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
            diff += random.gauss(0, self.game.cfg['NOISE']['angle'])
        return diff

cdef class DistanceSensorCy(SensorCy):
    """
    Distance from bot to the target in 'crows flight'.
    """
    
    def __init__(self,
                 GameCy game,  # Type not specified due to circular imports
                 int sensor_id=0):
        """
        :param game: Reference to the game in which the sensor is used
        :param sensor_id: Identification number for the sensor
        """
        super().__init__(game=game,
                         sensor_id=sensor_id)
    
    def __str__(self):
        return f"{D_SENSOR_DISTANCE}_{self.id:02d}"
    
    cdef float get_measure(self):
        """
        :return: Distance between target and robot's center coordinate, which is a float
        """
        cdef Vec2dCy start_p
        cdef Vec2dCy end_p
        cdef float distance
        
        # Calculations
        start_p = self.game.player.pos
        end_p = self.game.target
        distance = (start_p - end_p).get_length()
        if self.game.noise: distance += random.gauss(0, self.game.cfg['NOISE']['distance'])
        return distance

cdef class ProximitySensorCy(SensorCy):
    """
    The proximity sensor is attached to a bot's edge and measures the distance min(max_distance, object_distance). In
    other words, it returns the distance to an object in its path (only a straight line) if this distance is within a
    certain threshold, otherwise the maximum value will be returned.
    """
    
    def __init__(self,
                 GameCy game,  # Type not specified due to circular imports
                 int sensor_id=0,
                 float angle=0,
                 float pos_offset=0,
                 float max_dist=0):
        """
        :param game: Reference to the game in which the sensor is used
        :param sensor_id: Identification number for the sensor
        :param angle: Relative angle to the agent's center of mass and orientation
        :param pos_offset: Distance to the agent's center of mass and orientation
        :param max_dist: Maximum distance the sensor can reach, infinite if set to zero
        """
        if not max_dist: max_dist = game.cfg['SENSOR']['ray distance']
        super().__init__(game=game,
                         sensor_id=sensor_id,
                         angle=angle,
                         pos_offset=pos_offset,
                         max_dist=max_dist)
        self.start_pos = None  # Placeholder for start-point of proximity sensor
        self.end_pos = None  # Placeholder for end-point of proximity sensor
    
    def __str__(self):
        return f"{D_SENSOR_PROXIMITY}_{self.id:02d}"
    
    cdef float get_measure(self):
        """
        Get the distance to the closest wall. If all the walls are 'far enough', as determined by self.max_dist, then
        the maximum sensor-distance is returned.
        
        :return: Float expressing the distance to the closest wall, if there is any
        """
        cdef Vec2dCy normalized_offset
        cdef Line2dCy sensor_line
        cdef float closest_dist
        cdef Line2dCy wall
        cdef bint inter
        cdef Vec2dCy pos
        
        # Start and end point of ray
        normalized_offset = Vec2dCy(np.cos(self.game.player.angle + self.angle),
                                    np.sin(self.game.player.angle + self.angle))
        self.start_pos = self.game.player.pos + normalized_offset * self.pos_offset
        self.end_pos = self.game.player.pos + normalized_offset * (self.pos_offset + self.max_dist)
        sensor_line = Line2dCy(x=self.game.player.pos, y=self.end_pos)
        
        # Check if there is a wall intersecting with the sensor and return the closest distance to a wall
        closest_dist = self.max_dist
        for wall in self.game.walls:
            inter, pos = line_line_intersection_cy(sensor_line, wall)
            if inter:
                new_dist = (pos - self.start_pos).get_length()
                if closest_dist > new_dist:
                    self.end_pos = pos
                    closest_dist = new_dist
        
        if self.game.noise:
            closest_dist += random.gauss(0, self.game.cfg['NOISE']['proximity'])
        return closest_dist
