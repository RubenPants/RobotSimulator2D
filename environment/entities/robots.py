"""
robots.py

Robots used to manoeuvre around in the Game-environment.
"""
from environment.entities.sensors import AngularSensor, DistanceSensor, ProximitySensor
from utils.config import *
from utils.dictionary import *
from utils.myutils import drop, prep
from utils.vec2d import angle_to_vec, Vec2d


class FootBot:
    """
    The FootBot is the main bot used in this project. It is a simple circular robot with two wheels on its sides.
    """
    
    def __init__(self,
                 game,  # Type not specified due to circular imports
                 init_pos: Vec2d = None,
                 init_orient: float = 0,
                 r: int = BOT_RADIUS):
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
        self.pos = Vec2d(0, 0)  # Current position
        self.prev_pos = Vec2d(0, 0)  # Previous current position
        if init_pos:
            self.init_pos = init_pos  # Initial position
            self.pos.x = init_pos.x
            self.pos.y = init_pos.y
            self.prev_pos.x = init_pos.x
            self.prev_pos.y = init_pos.y
        else:
            self.init_pos = Vec2d(0, 0)
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
    
    def __str__(self, gen=None):
        return "foot_bot".format("_{gen:04d}".format(gen=gen) if gen else "")
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def drive(self, dt: float, lw: float, rw: float):
        """
        Update the robot's position and orientation based on the action of the wheels.
        
        :param dt: Delta time (must always be positioned first)
        :param lw: Speed of the left wheel, float [-1,1]
        :param rw: Speed of the right wheel, float [-1,1]
        """
        if TIME_ALL:
            prep(key="robot_drive", silent=True)
        
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
        if TIME_ALL:
            drop(key="robot_drive", silent=True)
    
    def get_sensor_readings(self):
        """
        :return: Dictionary of the current sensory-readings
        """
        if TIME_ALL:
            prep(key="sensor_readings", silent=True)
        
        # Collect all the sensor values in a dictionary
        sensor_readings = dict()
        sensor_readings[D_SENSOR_PROXIMITY] = self.get_sensor_reading_proximity()
        sensor_readings[D_SENSOR_DISTANCE] = self.get_sensor_reading_distance()
        sensor_readings[D_SENSOR_ANGLE] = self.get_sensor_reading_angle()
        
        if TIME_ALL:
            drop(key="sensor_readings", silent=True)
        
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
    
    def get_sensor_reading_angle(self):
        """
        :return: The current values of the proximity sensors
        """
        readings = dict()
        for a in self.angular_sensors:
            readings[a.id] = a.get_measure()
        return readings
    
    def get_sensor_reading_distance(self):
        """
        :return: The current distance to the target
        """
        return self.distance_sensor.get_measure()
    
    def get_sensor_reading_proximity(self):
        """
        :return: The current values of the proximity sensors
        """
        readings = dict()
        for p in self.proximity_sensors:
            readings[p.id] = p.get_measure()
        return readings
    
    # -----------------------------------------------> SENSOR METHODS <----------------------------------------------- #
    
    def add_angular_sensors(self, clockwise=True):
        """
        Add an angular sensor to the agent and give it an idea one greater than the last sensor added, or 0 if it is the
        first sensor that is added.
        """
        self.angular_sensors.add(AngularSensor(sensor_id=len(self.angular_sensors),
                                               game=self.game,
                                               clockwise=clockwise))
    
    def add_proximity_sensor(self, angle):
        """
        Add an proximity sensor to the agent and give it an idea one greater than the last sensor added, or 0 if it is
        the first sensor that is added.
        """
        self.proximity_sensors.add(ProximitySensor(sensor_id=len(self.proximity_sensors),
                                                   game=self.game,
                                                   angle=angle,
                                                   pos_offset=BOT_RADIUS))
    
    def create_angular_sensors(self):
        """
        Two angular sensors that define the angle between the orientation the agent is heading and the agent towards the
        target 'in crows flight'. One measures this angle in clockwise, the other counterclockwise.
        """
        self.add_angular_sensors(clockwise=True)
        self.add_angular_sensors(clockwise=False)
    
    def create_distance_sensor(self):
        """
        Single distance sensor which determines distance between agent's center and target's center.
        """
        self.distance_sensor = DistanceSensor(game=self.game)
    
    def create_proximity_sensors(self):
        """
        24 equally spaced proximity sensors, which measure the distance between the agent and an object, if this object
        is within 1.5 meters of distance.
        """
        for i in range(7):
            self.add_proximity_sensor(angle=-np.pi / 2 + i * np.pi / 6)
