"""
robots_cy.pyx

Robots used to manoeuvre around in the Game-environment.
"""
import numpy as np
cimport numpy as np

from sensors_cy cimport AngularSensorCy, DistanceSensorCy, ProximitySensorCy
from utils.cy.vec2d_cy cimport angle_to_vec, Vec2dCy

cdef class MarXBotCy:
    """The FootBot is the main bot used in this project. It is a simple circular robot with two wheels on its sides."""
    
    __slots__ = (
        "game",
        "pos", "prev_pos", "init_pos", "init_angle", "angle", "prev_angle", "radius",
        "sensors", "active_sensors",
    )
    
    def __init__(self,
                 GameCy game,
                 Vec2dCy init_pos=None,
                 float init_orient=0,
                 float r=0
                 ):
        """
        Create a new FootBot object.
        
        :param game: Reference to the game in which the robot is created [Game]
        :param init_pos: Initial position of the bot
        :param init_orient: Initial angle of the bot
        :param r: Radius of the circular robot
        """
        # Default values parameters
        if not r: r = game.bot_radius
        
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
        self.prev_angle = init_orient  # Previous angle
        self.radius = r  # Radius of the bot
        
        # Container of all the sensors
        self.sensors = dict()
        
        # Create the sensors (fixed order!)
        self.create_proximity_sensors()
        self.create_angular_sensors()
        self.add_distance_sensor()
        
        # Set all the active sensors
        self.active_sensors = set(self.sensors.keys())
    
    def __str__(self):
        return "foot_bot"
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    cpdef void drive(self, float dt, float lw, float rw):
        """
        Update the robot's position and orientation based on the action of the wheels.
        
        :param dt: Delta time (must always be positioned first)
        :param lw: Speed of the left wheel, float [-1,1]
        :param rw: Speed of the right wheel, float [-1,1]
        """
        # Constraint the inputs
        lw = max(min(lw, 1), -1)
        rw = max(min(rw, 1), -1)
        
        # Update previous state
        self.prev_pos.x, self.prev_pos.y = self.pos.x, self.pos.y
        self.prev_angle = self.angle
        
        # Update angle is determined by the speed of both wheels
        self.angle += (rw - lw) * self.game.bot_turning_speed * dt
        self.angle %= 2 * np.pi
        
        # Update position is the average of the two wheels times the maximum driving speed
        self.pos += angle_to_vec(self.angle) * float((((lw + rw) / 2) * self.game.bot_driving_speed * dt))
    
    cpdef list get_sensor_readings(self, set close_walls=None):
        """List of the current sensory-readings."""
        for i in self.active_sensors: self.sensors[i].measure(close_walls)
        return [self.sensors[i].value for i in sorted(self.active_sensors)]
    
    cpdef float get_sensor_readings_distance(self):
        """Value of current distance-reading."""
        return self.sensors[len(self.sensors) - 1].value  # Distance is always the last sensor
    
    def reset(self):
        """
        Put the robot back to its initial parameters.
        """
        self.pos.x = self.init_pos.x
        self.pos.y = self.init_pos.y
        self.prev_pos.x = self.init_pos.x
        self.prev_pos.y = self.init_pos.y
        self.angle = self.init_angle
    
    # -----------------------------------------------> SENSOR METHODS <----------------------------------------------- #
    
    cpdef void add_angular_sensors(self, bint clockwise=True):
        """
        Add an angular sensor to the agent and give it an idea one greater than the last sensor added, or 0 if it is the
        first sensor that is added.
        """
        self.sensors[len(self.sensors)] = AngularSensorCy(sensor_id=len(self.sensors),
                                                          game=self.game,
                                                          clockwise=clockwise)
    
    cpdef void add_distance_sensor(self):
        """Single distance sensor which determines distance between agent's center and target's center."""
        self.sensors[len(self.sensors)] = DistanceSensorCy(sensor_id=len(self.sensors),
                                                           game=self.game)
    
    cpdef void add_proximity_sensor(self, float angle):
        """
        Add an proximity sensor to the agent and give it an id one greater than the last sensor added, or 0 if it is
        the first sensor that is added.
        
        :param angle: Relative angle to the robot's facing-direction
                        * np.pi / 2 = 90° to the left of the robot
                        * 0 = the same direction as the robot is facing
                        * -np.pi / 2 = 90° to the right of the robot
        """
        self.sensors[len(self.sensors)] = ProximitySensorCy(sensor_id=len(self.sensors),
                                                            game=self.game,
                                                            angle=angle,
                                                            pos_offset=self.game.bot_radius)
    
    cpdef void create_angular_sensors(self):
        """
        Two angular sensors that define the angle between the orientation the agent is heading and the agent towards the
        target 'in crows flight'. One measures this angle in clockwise, the other counterclockwise.
        """
        self.add_angular_sensors(clockwise=True)
        self.add_angular_sensors(clockwise=False)
    
    cpdef void create_proximity_sensors(self):
        """
        13 proximity sensors, which measure the distance between the agent and an object, if this object is within 0.5
         meters of distance. The proximity sensors are not evenly spaced, since the fact that the robot has a front will
         be exploited. Sensors are added from the left-side of the drone to the right.
        """
        cdef int i
        self.add_proximity_sensor(angle=3 * np.pi / 4)  # -135°
        for i in range(5):  # -90° until -10° with hops of 20° (total of 5 sensors)
            self.add_proximity_sensor(angle=np.pi / 2 - i * np.pi / 9)
        self.add_proximity_sensor(angle=0)  # 0°
        for i in range(5):  # 10° until 90° with hops of 20° (total of 5 sensors)
            self.add_proximity_sensor(angle=-np.pi / 18 - i * np.pi / 9)
        self.add_proximity_sensor(angle=-3 * np.pi / 4)  # 135°
    
    cpdef list get_proximity_sensors(self):
        """Get a list of all proximity sensors."""
        return [self.sensors[i] for i in range(13)]
    
    cpdef void set_active_sensors(self, set connections):
        """
        Update all the sensor keys used by the robot.
        
        :param connections: Set of all connections in tuple format (sending node, receiving node)
        """
        self.active_sensors = get_active_sensors(connections=connections, total_input_size=len(self.sensors))
        

cpdef set get_active_sensors(set connections, int total_input_size):
    """Get a set of all the used input-sensors based on the connections. The distance sensor is always used."""
    # Exploit the fact that sensor inputs have negative connection keys
    used = {a + total_input_size for (a, _) in connections if a < 0}
    used.add(total_input_size - 1)  # Always use the distance sensor
    return used