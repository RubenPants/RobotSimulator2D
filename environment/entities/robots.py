"""
robots.py

Robots used to manoeuvre around in the Game-environment.
"""
from numpy import pi

from environment.entities.sensors import AngularSensor, DistanceSensor, ProximitySensor
from utils.vec2d import angle_to_vec, Vec2d


class MarXBot:
    """The FootBot is the main bot used in this project. It is a simple circular robot with two wheels on its sides."""
    
    __slots__ = (
        "game",
        "pos", "prev_pos", "init_pos", "init_angle", "angle", "prev_angle", "radius",
        "n_proximity", "n_angular", "n_distance",
        "sensors", "active_sensors",
    )
    
    def __init__(self,
                 game,  # Type not specified due to circular imports
                 r: float = 0
                 ):
        """
        Create a new FootBot object.
        
        :param game: Reference to the game in which the robot is created [Game]
        :param r: Radius of the circular robot
        """
        # Game specific parameter
        self.game = game  # Game in which robot runs
        
        # Robot specific parameters (Placeholders)
        self.pos = Vec2d(0, 0)  # Current position
        self.init_pos = Vec2d(0, 0)  # Initial position
        self.prev_pos = Vec2d(0, 0)  # Previous current position
        self.angle: float = 0  # Current angle
        self.init_angle: float = 0  # Initial angle
        self.prev_angle: float = 0  # Previous angle
        self.radius: float = r if r else game.bot_radius  # Radius of the bot
        
        # Container of all the sensors
        self.sensors: dict = dict()
        
        # Counters for number of sensors used
        self.n_proximity: int = 0
        self.n_angular: int = 0
        self.n_distance: int = 0
        
        # Create the sensors (fixed order!)
        self.create_proximity_sensors()
        self.create_angular_sensors()
        self.add_distance_sensor()
        
        # Number of distance-sensors must always be equal to 1
        assert self.n_distance == 1
        
        # Set all the sensors as active initially
        self.active_sensors = set(self.sensors.keys())
    
    def __str__(self):
        return "MarXBot"
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def drive(self, dt: float, lw: float, rw: float):
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
        self.angle %= 2 * pi
        
        # Update position is the average of the two wheels times the maximum driving speed
        self.pos += angle_to_vec(self.angle) * float((((lw + rw) / 2) * self.game.bot_driving_speed * dt))
    
    def get_sensor_readings(self, close_walls: set = None):
        """List of the current sensory-readings."""
        for i in self.active_sensors: self.sensors[i].measure(close_walls)
        return [self.sensors[i].value for i in sorted(self.active_sensors)]
    
    def get_sensor_readings_distance(self):
        """Value of current distance-reading."""
        return self.sensors[len(self.sensors) - 1].value  # Distance is always the last sensor
    
    def reset(self):
        """Put the robot back to its initial parameters."""
        self.pos.x = self.init_pos.x
        self.pos.y = self.init_pos.y
        self.prev_pos.x = self.init_pos.x
        self.prev_pos.y = self.init_pos.y
        self.angle = self.init_angle
        self.prev_angle = self.init_angle
    
    # -----------------------------------------------> SENSOR METHODS <----------------------------------------------- #
    
    def add_angular_sensors(self, clockwise: bool = True):
        """
        Add an angular sensor to the agent and give it an idea one greater than the last sensor added, or 0 if it is the
        first sensor that is added.
        """
        self.sensors[len(self.sensors)] = AngularSensor(sensor_id=len(self.sensors),
                                                        game=self.game,
                                                        clockwise=clockwise)
        self.n_angular += 1
    
    def add_distance_sensor(self):
        """Single distance sensor which determines distance between agent's center and target's center."""
        self.sensors[len(self.sensors)] = DistanceSensor(sensor_id=len(self.sensors),
                                                         game=self.game)
        self.n_distance += 1
    
    def add_proximity_sensor(self, angle: float):
        """
        Add an proximity sensor to the agent and give it an id one greater than the last sensor added, or 0 if it is
        the first sensor that is added.
        
        :param angle: Relative angle to the robot's facing-direction
                        * pi / 2 = 90° to the left of the robot
                        * 0 = the same direction as the robot is facing
                        * -pi / 2 = 90° to the right of the robot
        """
        self.sensors[len(self.sensors)] = ProximitySensor(sensor_id=len(self.sensors),
                                                          game=self.game,
                                                          angle=angle,
                                                          pos_offset=self.game.bot_radius,
                                                          max_dist=self.game.ray_distance,
                                                          )
        self.n_proximity += 1
    
    def create_angular_sensors(self):
        """
        Two angular sensors that define the angle between the orientation the agent is heading and the agent towards the
        target 'in crows flight'. One measures this angle in clockwise, the other counterclockwise.
        """
        for clockwise in get_angular_directions(): self.add_angular_sensors(clockwise=clockwise)
    
    def create_proximity_sensors(self):
        """
        13 proximity sensors, which measure the distance between the agent and an object, if this object is within 0.5
         meters of distance. The proximity sensors are not evenly spaced, since the fact that the robot has a front will
         be exploited. Sensors are added from the left-side of the drone to the right.
        """
        for angle in get_proximity_angles(): self.add_proximity_sensor(angle=angle)
    
    def set_active_sensors(self, connections: set):
        """
        Update all the sensor keys used by the robot.
        
        :param connections: Set of all connections in tuple format (sending node, receiving node)
        """
        self.active_sensors = get_active_sensors(connections=connections, total_input_size=len(self.sensors))


def get_active_sensors(connections: set, total_input_size: int):
    """Get a set of all the used input-sensors based on the connections. The distance sensor is always used."""
    # Exploit the fact that sensor inputs have negative connection keys
    used = {a + total_input_size for (a, _) in connections if a < 0}
    used.add(total_input_size - 1)  # Always use the distance sensor
    return used


def get_proximity_angles():
    """Get the angles used for the proximity sensors."""
    angles = []
    """
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
    # """
    return angles


def get_angular_directions():
    """Get the clockwise directions for the angular sensors."""
    return []
    # return [True, False]


def get_number_of_sensors():
    """Get the number of sensors mounted on the robot."""
    return len(get_proximity_angles()) + len(get_angular_directions()) + 1


def get_snapshot():
    """
    Get the snapshot of the current robot-configuration. This method mimics the 'Create the sensors' section in the
    MarXBot creation process.
    """
    sorted_names = []
    # Proximity sensors
    for a in get_proximity_angles(): sorted_names.append(str(ProximitySensor(game=None, max_dist=1, angle=a)))
    
    # Angular sensors
    for cw in get_angular_directions(): sorted_names.append(str(AngularSensor(game=None, clockwise=cw)))
    
    # Distance sensor
    sorted_names.append(str(DistanceSensor(game=None)))
    
    # Negate all the keys to create the snapshot
    snapshot = dict()
    for i, name in enumerate(sorted_names):
        snapshot[-len(sorted_names) + i] = name
    
    return snapshot
