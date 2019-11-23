"""
god_class_cy.pyx

To bypass the 'type not recognized' error, I copy pasted all the types within one God class and optimized them
correspondingly.
"""
import math
import operator
import pickle
import random

import pylab as pl
from matplotlib import collections as mc

from utils.config import *
from utils.dictionary import *

# ----------------------------------------------------> CONSTANTS <--------------------------------------------------- #

cdef float EPSILON = 1e-5

# ------------------------------------------------------> Vec2D <----------------------------------------------------- #

cdef class Vec2dCy:
    """
    Create a two dimensional vector.
    """
    cdef public float x, y
    
    __slots__ = ("x", "y")
    
    def __init__(self, float x=0, float y=0):
        self.x = x
        self.y = y
    
    def __getitem__(self, int i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        raise IndexError()
    
    def __setitem__(self, int i, float value):
        if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        else:
            raise IndexError()
    
    def __iter__(self):
        yield self.x
        yield self.y
    
    def __len__(self):
        return 2
    
    def __repr__(self):
        return 'Vec2dCy(%s, %s)' % (self.x, self.y)
    
    def __eq__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            return self.x == other[0] and self.y == other[1]
        else:
            return False
    
    def __ne__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            return self.x != other[0] or self.y != other[1]
        else:
            return True
    
    def __add__(self, other):
        if isinstance(other, Vec2dCy):
            return Vec2dCy(self.x + other.x, self.y + other.y)
        elif hasattr(other, "__getitem__"):
            return Vec2dCy(self.x + other[0], self.y + other[1])
        else:
            return Vec2dCy(self.x + other, self.y + other)
    
    __radd__ = __add__
    
    def __iadd__(self, other):
        if isinstance(other, Vec2dCy):
            self.x += other.x
            self.y += other.y
        elif hasattr(other, "__getitem__"):
            self.x += other[0]
            self.y += other[1]
        else:
            self.x += other
            self.y += other
        return self
    
    def __sub__(self, other):
        if isinstance(other, Vec2dCy):
            return Vec2dCy(self.x - other.x, self.y - other.y)
        elif hasattr(other, "__getitem__"):
            return Vec2dCy(self.x - other[0], self.y - other[1])
        else:
            return Vec2dCy(self.x - other, self.y - other)
    
    def __rsub__(self, other):
        if isinstance(other, Vec2dCy):
            return Vec2dCy(other.x - self.x, other.y - self.y)
        if hasattr(other, "__getitem__"):
            return Vec2dCy(other[0] - self.x, other[1] - self.y)
        else:
            return Vec2dCy(other - self.x, other - self.y)
    
    def __isub__(self, other):
        if isinstance(other, Vec2dCy):
            self.x -= other.x
            self.y -= other.y
        elif hasattr(other, "__getitem__"):
            self.x -= other[0]
            self.y -= other[1]
        else:
            self.x -= other
            self.y -= other
        return self
    
    def __mul__(self, other):
        if isinstance(other, Vec2dCy):
            return Vec2dCy(self.x * other.x, self.y * other.y)
        if hasattr(other, "__getitem__"):
            return Vec2dCy(self.x * other[0], self.y * other[1])
        else:
            return Vec2dCy(self.x * other, self.y * other)
    
    __rmul__ = __mul__
    
    def __imul__(self, other):
        if isinstance(other, Vec2dCy):
            self.x *= other.x
            self.y *= other.y
        elif hasattr(other, "__getitem__"):
            self.x *= other[0]
            self.y *= other[1]
        else:
            self.x *= other
            self.y *= other
        return self
    
    cpdef Vec2dCy _operator_handler(self, other, f):
        if isinstance(other, Vec2dCy):
            return Vec2dCy(f(self.x, other.x),
                           f(self.y, other.y))
        elif hasattr(other, "__getitem__"):
            return Vec2dCy(f(self.x, other[0]),
                           f(self.y, other[1]))
        else:
            return Vec2dCy(f(self.x, other),
                           f(self.y, other))
    
    cpdef Vec2dCy _right_operator_handler(self, other, f):
        if hasattr(other, "__getitem__"):
            return Vec2dCy(f(other[0], self.x),
                           f(other[1], self.y))
        else:
            return Vec2dCy(f(other, self.x),
                           f(other, self.y))
    
    cpdef Vec2dCy _inplace_operator_handler(self, other, f):
        if hasattr(other, "__getitem__"):
            self.x = f(self.x, other[0])
            self.y = f(self.y, other[1])
        else:
            self.x = f(self.x, other)
            self.y = f(self.y, other)
        return self
    
    def __div__(self, other):
        return self._operator_handler(other, operator.div)
    
    def __rdiv__(self, other):
        return self._right_operator_handler(other, operator.div)
    
    def __idiv__(self, other):
        return self._inplace_operator_handler(other, operator.div)
    
    def __floordiv__(self, other):
        return self._operator_handler(other, operator.floordiv)
    
    def __rfloordiv__(self, other):
        return self._right_operator_handler(other, operator.floordiv)
    
    def __ifloordiv__(self, other):
        return self._inplace_operator_handler(other, operator.floordiv)
    
    def __truediv__(self, other):
        return self._operator_handler(other, operator.truediv)
    
    def __rtruediv__(self, other):
        return self._right_operator_handler(other, operator.truediv)
    
    def __itruediv__(self, other):
        return self._inplace_operator_handler(other, operator.truediv)
    
    def __neg__(self):
        return Vec2dCy(operator.neg(self.x), operator.neg(self.y))
    
    def __pos__(self):
        return Vec2dCy(operator.pos(self.x), operator.pos(self.y))
    
    def __abs__(self):
        return Vec2dCy(abs(self.x), abs(self.y))
    
    def __invert__(self):
        return Vec2dCy(-self.x, -self.y)
    
    cpdef float get_angle(self):
        if self.get_length() == 0:
            return 0
        return math.atan2(self.y, self.x)
    
    cpdef float get_length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    cpdef Vec2dCy normalized(self):
        length = self.get_length()
        if length != 0:
            return self / length
        return Vec2dCy(self)

cpdef Vec2dCy angle_to_vec(float angle):
    """
    Transform an angle to a normalized vector.

    :param angle: Float
    :return: Vec2dCy
    """
    return Vec2dCy(np.cos(angle), np.sin(angle))

# -----------------------------------------------------> Line2D <----------------------------------------------------- #

cdef class Line2dCy:
    """
    Create a two dimensional line setup of the connection between two 2D vectors.
    """
    cdef public Vec2dCy x, y
    
    __slots__ = ("x", "y")
    
    def __init__(self, Vec2dCy x=None, Vec2dCy y=None):
        self.x = x if x else Vec2dCy(0, 0)
        self.y = y if y else Vec2dCy(0, 0)
    
    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        raise IndexError()
    
    def __setitem__(self, int i, Vec2dCy value):
        if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        else:
            raise IndexError()
    
    def __iter__(self):
        yield self.x
        yield self.y
    
    def __len__(self):
        return 2
    
    def __repr__(self):
        return 'Line2dCy(%s, %s)' % (self.x, self.y)
    
    def __eq__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            return self.x == other[0] and self.y == other[1]
        else:
            return False
    
    def __ne__(self, other):
        if hasattr(other, "__getitem__") and len(other) == 2:
            return self.x != other[0] or self.y != other[1]
        else:
            return True
    
    def __add__(self, other):
        if isinstance(other, Line2dCy):
            return Line2dCy(self.x + other.x, self.y + other.y)
        elif hasattr(other, "__getitem__"):
            return Line2dCy(self.x + other[0], self.y + other[1])
        else:
            return Line2dCy(self.x + other, self.y + other)
    
    __radd__ = __add__
    
    def __iadd__(self, other):
        if isinstance(other, Line2dCy):
            self.x += other.x
            self.y += other.y
        elif hasattr(other, "__getitem__"):
            self.x += other[0]
            self.y += other[1]
        else:
            self.x += other
            self.y += other
        return self
    
    def __sub__(self, other):
        if isinstance(other, Line2dCy):
            return Line2dCy(self.x - other.x, self.y - other.y)
        elif hasattr(other, "__getitem__"):
            return Line2dCy(self.x - other[0], self.y - other[1])
        else:
            return Line2dCy(self.x - other, self.y - other)
    
    def __rsub__(self, other):
        if isinstance(other, Line2dCy):
            return Line2dCy(other.x - self.x, other.y - self.y)
        if hasattr(other, "__getitem__"):
            return Line2dCy(other[0] - self.x, other[1] - self.y)
        else:
            return Line2dCy(other - self.x, other - self.y)
    
    def __isub__(self, other):
        if isinstance(other, Line2dCy):
            self.x -= other.x
            self.y -= other.y
        elif hasattr(other, "__getitem__"):
            self.x -= other[0]
            self.y -= other[1]
        else:
            self.x -= other
            self.y -= other
        return self
    
    cpdef float get_length(self):
        return (self.x - self.y).get_length()
    
    cpdef float get_orientation(self):
        """
        Get the orientation from start to end.
        """
        return (self.x - self.y).get_angle()

# --------------------------------------------------> INTERSECTION <-------------------------------------------------- #


cpdef line_line_intersection_cy(Line2dCy l1, Line2dCy l2):
    """
    Determine if two lines are intersecting with each other and give point of contact if they do.
    
    :param l1: Line2d
    :param l2: Line2d
    :return: Bool, Intersection: Vec2d
    """
    # Define used parameters
    cdef float a_dev
    cdef float a
    cdef float b_dev
    cdef float b
    
    # Calculations
    a_dev = ((l2.y.y - l2.x.y) * (l1.y.x - l1.x.x) - (l2.y.x - l2.x.x) * (l1.y.y - l1.x.y))
    a_dev = a_dev if a_dev != 0 else EPSILON
    a = ((l2.y.x - l2.x.x) * (l1.x.y - l2.x.y) - (l2.y.y - l2.x.y) * (l1.x.x - l2.x.x)) / a_dev
    b_dev = ((l2.y.y - l2.x.y) * (l1.y.x - l1.x.x) - (l2.y.x - l2.x.x) * (l1.y.y - l1.x.y))
    b_dev = b_dev if b_dev != 0 else EPSILON
    b = ((l1.y.x - l1.x.x) * (l1.x.y - l2.x.y) - (l1.y.y - l1.x.y) * (l1.x.x - l2.x.x)) / b_dev
    
    # Check if not intersecting
    if 0 <= a <= 1 and 0 <= b <= 1:
        return True, Vec2dCy(l1.x.x + (a * (l1.y.x - l1.x.x)), l1.x.y + (a * (l1.y.y - l1.x.y)))
    else:
        return False, None

cpdef bint point_circle_intersection_cy(Vec2dCy p, Vec2dCy c, float r):
    """
    Determine if a point lays inside of a circle.
    
    :param p: Point
    :param c: Center of circle
    :param r: Radius of circle
    :return: Bool
    """
    return (p - c).get_length() < r + EPSILON

cpdef bint point_line_intersection_cy(Vec2dCy p, Line2dCy l):
    """
    Determine if a point lays on a line.
    
    :param p: Point
    :param l: Line2d
    :return: Bool
    """
    return l.get_length() - EPSILON <= (p - l.x).get_length() + (p - l.y).get_length() <= l.get_length() + EPSILON

cpdef circle_line_intersection_cy(Vec2dCy c, float r, Line2dCy l):
    """
    Determine if a circle intersects with a line and give point of contact if they do.
    
    :param l: Line2d
    :param c: Center of circle
    :param r: Radius of circle
    :return: Bool, Intersection: Vec2d
    """
    # Define used parameters
    cdef float dot
    cdef Vec2dCy closest
    
    # Check for the edges of the line
    if point_circle_intersection_cy(l.x, c, r):
        return True, l.x
    if point_circle_intersection_cy(l.y, c, r):
        return True, l.y
    
    # Determine closest point to the line
    dot = (((c.x - l.x.x) * (l.y.x - l.x.x)) + ((c.y - l.x.y) * (l.y.y - l.x.y))) / (l.get_length() ** 2)
    closest = Vec2dCy(l.x.x + (dot * (l.y.x - l.x.x)), l.x.y + (dot * (l.y.y - l.x.y)))
    
    # Check if closest is on segment
    if not point_line_intersection_cy(p=closest, l=l):
        return False, None
    
    # Check if in circle
    return (True, closest) if (closest - c).get_length() <= (r + EPSILON) else (False, None)

# -----------------------------------------------------> SENSORS <---------------------------------------------------- #

cdef class SensorCy:
    """
    The baseclass used by all sensors.
    """
    cdef public GameCy game
    cdef public int id
    cdef public float angle
    cdef public float pos_offset
    cdef public float max_dist
    
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
        """
        :return: Name of the sensor
        """
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
    cdef public bint clockwise
    
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
        super().__init__(game=game,
                         sensor_id=sensor_id)
        self.clockwise = clockwise
    
    def __str__(self):
        return "{sensor}_{id:02d}".format(sensor=D_SENSOR_ANGLE, id=self.id)
    
    cdef float get_measure(self):
        """
        :return: Float between 0 and 2*PI
        """
        # Define used parameters
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
            diff += random.gauss(0, NOISE_SENSOR_ANGLE)
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
        return "{sensor}_{id:02d}".format(sensor=D_SENSOR_DISTANCE, id=self.id)
    
    cdef float get_measure(self):
        """
        :return: Distance between target and robot's center coordinate, which is a float
        """
        # Define used parameters
        cdef Vec2dCy start_p
        cdef Vec2dCy end_p
        cdef float distance
        
        # Calculations
        start_p = self.game.player.pos
        end_p = self.game.target
        distance = (start_p - end_p).get_length()
        if self.game.noise:
            distance += random.gauss(0, NOISE_SENSOR_DIST)
        return distance

cdef class ProximitySensorCy(SensorCy):
    """
    The proximity sensor is attached to a bot's edge and measures the distance min(max_distance, object_distance). In
    other words, it returns the distance to an object in its path (only a straight line) if this distance is within a
    certain threshold, otherwise the maximum value will be returned.
    """
    cdef public Vec2dCy end_pos
    
    def __init__(self,
                 GameCy game,  # Type not specified due to circular imports
                 int sensor_id=0,
                 float angle=0,
                 float pos_offset=0,
                 float max_dist=SENSOR_RAY_DISTANCE):
        """
        :param game: Reference to the game in which the sensor is used
        :param sensor_id: Identification number for the sensor
        :param angle: Relative angle to the agent's center of mass
        :param pos_offset: Distance to the agent's center of mass
        :param max_dist: Maximum distance the sensor can reach, infinite if set to zero
        """
        super().__init__(game=game,
                         sensor_id=sensor_id,
                         angle=angle,
                         pos_offset=pos_offset,
                         max_dist=max_dist)
        self.end_pos = None  # Placeholder for end-point of proximity sensor
    
    def __str__(self):
        return "{sensor}_{id:02d}".format(sensor=D_SENSOR_PROXIMITY, id=self.id)
    
    cdef float get_measure(self):
        """
        Get the distance to the closest wall. If all the walls are 'far enough', as determined by self.max_dist, then
        the maximum sensor-distance is returned.
        
        :return: Float expressing the distance to the closest wall, if there is any
        """
        # Define used parameters
        cdef Vec2dCy normalized_offset
        cdef Line2dCy sensor_line
        cdef float closest_dist
        cdef Line2dCy wall
        cdef bint inter
        cdef Vec2dCy pos
        
        # Start and end point of ray
        normalized_offset = Vec2dCy(np.cos(self.game.player.angle + self.angle),
                                    np.sin(self.game.player.angle + self.angle))
        self.end_pos = self.game.player.pos + normalized_offset * (self.pos_offset + self.max_dist)
        sensor_line = Line2dCy(x=self.game.player.pos,
                               y=self.end_pos)
        
        # Check if there is a wall intersecting with the sensor and return the closest distance to a wall
        closest_dist = self.max_dist
        for wall in self.game.walls:
            inter, pos = line_line_intersection_cy(sensor_line, wall)
            if inter:
                new_dist = (pos - self.game.player.pos).get_length()
                if closest_dist > new_dist:
                    self.end_pos = pos
                    closest_dist = new_dist
        
        if self.game.noise:
            closest_dist += random.gauss(0, NOISE_SENSOR_PROXY)
        return closest_dist

# ------------------------------------------------------> ROBOT <----------------------------------------------------- #

cdef class FootBotCy:
    """
    The FootBot is the main bot used in this project. It is a simple circular robot with two wheels on its sides.
    """
    cdef public GameCy game
    cdef public Vec2dCy pos
    cdef public Vec2dCy prev_pos
    cdef public Vec2dCy init_pos
    cdef public float init_angle
    cdef public float angle
    cdef public float radius
    cdef public set angular_sensors
    cdef public DistanceSensorCy distance_sensor
    cdef public set proximity_sensors
    
    def __init__(self,
                 GameCy game,  # Type not specified due to circular imports
                 Vec2dCy init_pos=None,
                 float init_orient=0,
                 float r=BOT_RADIUS):
        """
        Create a new FootBot object.
        
        :param game: Reference to the game in which the robot is created [Game]
        :param init_pos: Initial position of the bot
        :param init_orient: Initial angle of the bot
        :param r: Radius of the circular robot
        """
        # Game specific parameter
        self.game = game  # Game in which robot runs
        
        # Robot specific parameters
        self.pos = Vec2dCy(0, 0)  # Current position
        self.prev_pos = Vec2dCy(0, 0)  # Previous current position
        if init_pos:
            self.init_pos = init_pos  # Initial position
            self.pos.x = init_pos.x
            self.pos.y = init_pos.y
            self.prev_pos.x = init_pos.x
            self.prev_pos.y = init_pos.y
        else:
            self.init_pos = Vec2dCy(0, 0)
        self.init_angle = init_orient  # Initial angle
        self.angle = init_orient  # Current angle
        self.radius = r  # Radius of the bot
        
        # Placeholders for sensors
        self.angular_sensors = set()
        self.distance_sensor = None
        self.proximity_sensors = set()
        
        # Create the sensors
        self.create_angular_sensors()
        self.create_distance_sensor()
        self.create_proximity_sensors()
    
    def __str__(self):
        return "foot_bot"
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    cdef void drive(self, float dt, float lw, float rw):
        """
        Update the robot's position and orientation based on the action of the wheels.
        
        :param dt: Delta time (must always be positioned first)
        :param lw: Speed of the left wheel, float [-1,1]
        :param rw: Speed of the right wheel, float [-1,1]
        """
        # Constraint the inputs
        lw = max(min(lw, 1), -1)
        rw = max(min(rw, 1), -1)
        
        # Update previous position
        self.prev_pos.x, self.prev_pos.y = self.pos.x, self.pos.y
        
        # Update angle is determined by the speed of both wheels
        self.angle += (rw - lw) * BOT_TURNING_SPEED * dt
        self.angle %= 2 * np.pi
        
        # Update position is the average of the two wheels times the maximum driving speed
        self.pos += angle_to_vec(self.angle) * float((((lw + rw) / 2) * BOT_DRIVING_SPEED * dt))
    
    cdef dict get_sensor_readings(self):
        """
        :return: Dictionary of the current sensory-readings
        """
        # Collect all the sensor values in a dictionary
        cdef dict sensor_readings = dict()
        sensor_readings[D_SENSOR_PROXIMITY] = self.get_sensor_reading_proximity()
        sensor_readings[D_SENSOR_DISTANCE] = self.get_sensor_reading_distance()
        sensor_readings[D_SENSOR_ANGLE] = self.get_sensor_reading_angle()
        return sensor_readings
    
    def reset(self):
        """
        Put the robot back to its initial parameters.
        """
        self.pos.x = self.init_pos.x
        self.pos.y = self.init_pos.y
        self.prev_pos.x = self.init_pos.x
        self.prev_pos.y = self.init_pos.y
        self.angle = self.init_angle
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    cdef dict get_sensor_reading_angle(self):
        """
        :return: The current values of the proximity sensors
        """
        cdef dict readings = dict()
        cdef AngularSensorCy a
        
        for a in self.angular_sensors:
            readings[a.id] = a.get_measure()
        return readings
    
    cdef float get_sensor_reading_distance(self):
        """
        :return: The current distance to the target
        """
        return self.distance_sensor.get_measure()
    
    cdef dict get_sensor_reading_proximity(self):
        """
        :return: The current values of the proximity sensors
        """
        cdef dict readings = dict()
        cdef ProximitySensorCy p
        
        for p in self.proximity_sensors:
            readings[p.id] = p.get_measure()
        return readings
    
    # -----------------------------------------------> SENSOR METHODS <----------------------------------------------- #
    
    cdef void add_angular_sensors(self, bint clockwise=True):
        """
        Add an angular sensor to the agent and give it an idea one greater than the last sensor added, or 0 if it is the
        first sensor that is added.
        """
        self.angular_sensors.add(AngularSensorCy(sensor_id=len(self.angular_sensors),
                                                 game=self.game,
                                                 clockwise=clockwise))
    
    cdef void add_proximity_sensor(self, float angle):
        """
        Add an proximity sensor to the agent and give it an id one greater than the last sensor added, or 0 if it is
        the first sensor that is added.
        
        :param angle: Relative angle to the robot's facing-direction
                        * np.pi / 2 = 90° to the left of the robot
                        * 0 = the same direction as the robot is facing
                        * -np.pi / 2 = 90° to the right of the robot
        """
        self.proximity_sensors.add(ProximitySensorCy(sensor_id=len(self.proximity_sensors),
                                                     game=self.game,
                                                     angle=angle,
                                                     pos_offset=BOT_RADIUS))
    
    cdef void create_angular_sensors(self):
        """
        Two angular sensors that define the angle between the orientation the agent is heading and the agent towards the
        target 'in crows flight'. One measures this angle in clockwise, the other counterclockwise.
        """
        self.add_angular_sensors(clockwise=True)
        self.add_angular_sensors(clockwise=False)
    
    cdef void create_distance_sensor(self):
        """
        Single distance sensor which determines distance between agent's center and target's center.
        """
        self.distance_sensor = DistanceSensorCy(game=self.game)
    
    cdef void create_proximity_sensors(self):
        """
        24 equally spaced proximity sensors, which measure the distance between the agent and an object, if this object
        is within 1.5 meters of distance.
        """
        cdef int i
        
        for i in range(7):
            self.add_proximity_sensor(angle=-np.pi / 2 + i * np.pi / 6)

# ------------------------------------------------------> GAME <------------------------------------------------------ #

cdef class GameCy:
    """
    A game environment is built up from the following segments:
        * walls: Set of Line2d objects representing the walls in the maze
        * robot: The player manoeuvring in the environment
        * target: Robot that must be reached by the robot
    """
    cdef public str rel_path
    cdef public bint silent
    cdef public bint noise
    cdef public int id
    cdef public FootBotCy player
    cdef public Vec2dCy target
    cdef public list walls
    cdef public bint done
    
    def __init__(self,
                 int game_id=0,
                 bint noise=True,
                 bint overwrite=False,
                 str rel_path='environment/',
                 bint silent=False):
        """
        Define a new game.

        :param game_id: Game id
        :param noise: Add noise when progressing the game
        :param overwrite: Overwrite pre-existing games
        :param rel_path: Relative path where Game object is stored or will be stored
        :param silent: Do not print anything
        """
        # Set path correct
        self.rel_path = rel_path  # Relative path to the 'environment' folder
        self.silent = silent  # True: Do not print out statistics
        
        # Environment specific parameters
        self.noise = noise  # Add noise to the game-environment
        
        # Placeholders for parameters
        self.id = game_id  # Game's ID-number
        self.player = None  # Candidate-robot
        self.target = None  # Target-robot
        self.walls = None  # List of all walls in the game
        self.done = False  # Game has finished
        
        # Check if game already exists, if not create new game
        if overwrite or not self.load():
            self.create_empty_game()
    
    def __str__(self):
        return "game_{id:05d}".format(id=self.id)
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    cpdef dict close(self):
        """
        :return: Final state and useful statistics
        """
        return {
            D_GAME_ID:        self.id,
            D_POS:            self.player.pos,
            D_DIST_TO_TARGET: self.player.get_sensor_reading_distance(),
        }
    
    cpdef dict reset(self):
        """
        Reset the game.

        :return: Observation
        """
        self.player.reset()
        return self.get_observation()
    
    cpdef step(self, float l, float r):
        """
        Progress one step in the game.

        :param l: Left wheel speed [-1..1]
        :param r: Right wheel speed [-1..1]
        :return: Observation (Dictionary), target_reached (Boolean)
        """
        # Define used parameters
        cdef float dt
        cdef Line2dCy wall
        cdef bint inter
        cdef dict obs
        
        # Progress the game
        dt = 1.0 / FPS + abs(random.gauss(0, NOISE_TIME)) if self.noise else 1.0 / FPS
        self.player.drive(dt, lw=l, rw=r)
        
        # Check if intersected with a wall, if so then set player back to old position
        for wall in self.walls:
            inter, _ = circle_line_intersection_cy(c=self.player.pos, r=self.player.radius, l=wall)
            if inter:
                self.player.pos.x = self.player.prev_pos.x
                self.player.pos.y = self.player.prev_pos.y
                break
        
        # Get the current observations
        obs = self.get_observation()
        
        # Check if target reached
        if obs[D_DIST_TO_TARGET] <= TARGET_REACHED:
            self.done = True
        
        return obs, self.done
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    cpdef void create_empty_game(self):
        """
        Create an empty game that only contains the boundary walls.
        """
        # Create random set of walls
        self.walls = get_boundary_walls()
        self.target = Vec2dCy(0.5, AXIS_Y - 0.5)
        self.player = FootBotCy(game=self,
                                init_pos=Vec2dCy(AXIS_X - 0.5, 0.5),
                                init_orient=np.pi / 2)
        
        # Save the new game
        self.save()
        
        if not self.silent:
            print("New game created under id: {}".format(self.id))
    
    cpdef dict get_observation(self):
        """
        Get the current observation of the game. The following gets returned as a dictionary:
         * D_ANGLE: The angle the player is currently heading
         * D_DIST_TO_TARGET: Distance from player's current position to target in crows flight
         * D_GAME_ID: The game's ID
         * D_POS: The current position of the player in the maze (expressed in pixels)
         * D_SENSOR_LIST: List of all the sensors (proximity, angular, distance)
        """
        return {
            D_ANGLE:          self.player.angle,
            D_DIST_TO_TARGET: self.player.get_sensor_reading_distance(),
            D_GAME_ID:        self.id,
            D_POS:            self.player.pos,
            D_SENSOR_LIST:    self.get_sensor_list(),
        }
    
    cpdef list get_sensor_list(self):
        """
        Return a list of sensory-readings, with first the proximity sensors, then the angular sensors and at the end the
        distance-sensor.
        """
        cdef dict sensor_readings
        cdef dict proximity
        cdef dict angular
        cdef float distance
        cdef list result
        cdef int i
        
        # Read the sensors
        sensor_readings = self.player.get_sensor_readings()
        proximity = sensor_readings[D_SENSOR_PROXIMITY]
        angular = sensor_readings[D_SENSOR_ANGLE]
        distance = sensor_readings[D_SENSOR_DISTANCE]
        
        # Add sensory-readings in one list
        result = []
        for i in range(len(proximity)):  # Proximity IDs go from 0 to proximity_length
            result.append(proximity[i])
        for i in range(len(angular)):  # Angular IDs go from 0 to angular_length
            result.append(angular[i])
        result.append(distance)
        return result
    
    cpdef void set_player_angle(self, float a):
        """
        Set a new initial angle for the player.
        """
        self.player.init_angle = a
        self.player.angle = a
    
    cpdef void set_player_pos(self, Vec2dCy p):
        """
        Set a new initial position for the player.
        """
        self.player.init_pos.x = p.x
        self.player.init_pos.y = p.y
        self.player.pos.x = p.x
        self.player.pos.y = p.y
        self.player.prev_pos.x = p.x
        self.player.prev_pos.y = p.y
    
    # ---------------------------------------------> FUNCTIONAL METHODS <--------------------------------------------- #
    
    cpdef void save(self):
        cdef dict persist_dict = dict()
        persist_dict[D_PLAYER] = self.player
        persist_dict[D_TARGET] = self.target
        persist_dict[D_WALLS] = self.walls
        with open('{p}games_db/{g}'.format(p=self.rel_path, g=self), 'wb') as f:
            pickle.dump(persist_dict, f)
    
    cpdef bint load(self):
        """
        Load in a game, specified by its current id.

        :return: True: game successfully loaded | False: otherwise
        """
        # Define used parameter
        cdef dict game
        
        try:
            with open('{p}games_db/{g}'.format(p=self.rel_path, g=self), 'rb') as f:
                game = pickle.load(f)
            self.player = game[D_PLAYER]
            self.target = game[D_TARGET]
            self.walls = game[D_WALLS]
            if not self.silent:
                print("Existing game loaded with id: {}".format(self.id))
            return True
        except FileNotFoundError:
            return False
    
    cpdef get_blueprint(self):
        """
        :return: The blue-print map of the board (matplotlib Figure)
        """
        # Define used parameters
        cdef list walls
        cdef Line2dCy w
        
        fig, ax = pl.subplots()
        
        # Draw all the walls
        walls = []
        for w in self.walls:
            walls.append([(w.x.x, w.x.y), (w.y.x, w.y.y)])
        lc = mc.LineCollection(walls, linewidths=2)
        ax.add_collection(lc)
        
        # Add target to map
        pl.plot(0.5, AXIS_Y - 0.5, 'go')
        
        # Adjust the boundaries
        pl.xlim(0, AXIS_X)
        pl.ylim(0, AXIS_Y)
        
        # Return the figure in its current state
        return ax

cpdef list get_boundary_walls():
    """
    :return: Set of the four boundary walls
    """
    a = Vec2dCy(0, 0)
    b = Vec2dCy(AXIS_X, 0)
    c = Vec2dCy(AXIS_X, AXIS_Y)
    d = Vec2dCy(0, AXIS_Y)
    return [Line2dCy(a, b), Line2dCy(b, c), Line2dCy(c, d), Line2dCy(d, a)]
