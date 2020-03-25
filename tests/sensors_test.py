"""
sensor_test.py

Test of all the sensors.
"""
import os
import unittest

import numpy as np

from config import Config
from environment.entities.game import Game
from environment.entities.sensors import ProximitySensor
from utils.dictionary import D_SENSOR_LIST
from utils.line2d import Line2d
from utils.vec2d import Vec2d

# Parameters
RAY_DISTANCE = 2  # Maximum range of the proximity-sensor
EPSILON_ANGLE = 0.0001  # 0.0001 radian offset allowed (~0.02 degrees)
EPSILON_DISTANCE = 0.001  # 1 millimeter offset allowed
EPSILON_DISTANCE_L = 0.1  # 10 centimeter offset allowed


class AngularSensorTest(unittest.TestCase):
    """Test the angular sensor."""
    
    def test_front(self):
        """> Test angular sensors when target straight in the front."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2d(1, 0)
        game.set_player_init_pos(Vec2d(0, 0))
        game.set_player_init_angle(0)
        
        # Update the player's sensor-set
        game.player.sensors = dict()
        game.player.add_angular_sensors(clockwise=True)
        game.player.add_angular_sensors(clockwise=False)
        game.player.active_sensors = set(game.player.sensors.keys())
        
        # The third and second last sensors
        sensor_values = game.player.get_sensor_readings()
        self.assertEqual(len(sensor_values), 2)
        for s in sensor_values:
            s = min(s, abs(s - 2 * np.pi))
            self.assertAlmostEqual(s, 0.0, delta=EPSILON_ANGLE)
    
    def test_left_angle(self):
        """> Test the angular sensors when target on the left."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2d(1, 1)
        game.set_player_init_pos(Vec2d(0, 0))
        game.set_player_init_angle(0)  # Looking to the right
        
        # Update the player's sensor-set
        game.player.sensors = dict()
        game.player.add_angular_sensors(clockwise=True)
        game.player.add_angular_sensors(clockwise=False)
        game.player.active_sensors = set(game.player.sensors.keys())
        
        sensor_values = game.player.get_sensor_readings()
        self.assertEqual(len(sensor_values), 2)
        self.assertAlmostEqual(sensor_values[0], 7 * np.pi / 4, delta=EPSILON_ANGLE)  # Clockwise
        self.assertAlmostEqual(sensor_values[1], np.pi / 4, delta=EPSILON_ANGLE)  # Anti-clockwise


class DistanceSensorTest(unittest.TestCase):
    """Test the distance sensor."""
    
    def test_front(self):
        """> Test the distance sensor when target straight in the front."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2d(1, 0)
        game.set_player_init_pos(Vec2d(0, 0))
        game.set_player_init_angle(0)
        
        # Ask for the distance
        game.get_observation()
        self.assertAlmostEqual(game.player.get_sensor_readings_distance(), 1.0, delta=EPSILON_DISTANCE)
    
    def test_left_angle(self):
        """> Test distance sensor when target under an angle (towards the left)."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2d(1, 1)
        game.set_player_init_pos(Vec2d(0, 0))
        game.set_player_init_angle(0)
        
        # Ask for the distance
        game.get_observation()
        self.assertAlmostEqual(game.player.get_sensor_readings_distance(), np.sqrt(2), delta=EPSILON_DISTANCE)


class DeltaDistanceSensorTest(unittest.TestCase):
    """Test the delta distance sensor."""
    
    def test_front(self):
        """> Test the distance sensor when target straight in the front."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2d(1, 0)
        game.set_player_init_pos(Vec2d(0, 0))
        game.set_player_init_angle(0)
        
        # Update the sensors used to only include the delta-distance sensor
        game.player.sensors = dict()
        game.player.add_delta_distance_sensor()
        game.player.active_sensors = set(game.player.sensors.keys())
        
        # Initially the sensor should read zero
        sensor_values = game.player.get_sensor_readings()
        self.assertAlmostEqual(sensor_values[0], 0.0, delta=EPSILON_DISTANCE)
        
        # Advance the player's position by 0.1 meters and test sensor-reading
        game.player.pos = Vec2d(0.1, 0)
        sensor_values = game.player.get_sensor_readings()
        self.assertAlmostEqual(sensor_values[0], 0.1, delta=EPSILON_DISTANCE)
        
        # Advance the player's position by 0.001 meters backwards and test sensor-reading
        game.player.pos = Vec2d(0.0999999, 0)
        sensor_values = game.player.get_sensor_readings()
        self.assertAlmostEqual(sensor_values[0], -0.0000001, delta=EPSILON_DISTANCE)
    
    def test_equal_side(self):
        """> Test distance sensor when target on the sides with equal distance."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2d(1, 1)
        game.set_player_init_pos(Vec2d(0.999, 0))
        game.set_player_init_angle(0)
        
        # Update the sensors used to only include the delta-distance sensor
        game.player.sensors = dict()
        game.player.add_delta_distance_sensor()
        game.player.active_sensors = set(game.player.sensors.keys())
        
        # Initially the sensor should read zero
        sensor_values = game.player.get_sensor_readings()
        self.assertAlmostEqual(sensor_values[0], 0.0, delta=EPSILON_DISTANCE)
        
        # Advance the player's position to a symmetric position with equal distance
        game.player.pos = Vec2d(1.001, 0)
        sensor_values = game.player.get_sensor_readings()
        self.assertAlmostEqual(sensor_values[0], 0.0, delta=EPSILON_DISTANCE)
    
    def test_none_zero_drive(self):
        """Test if the delta-distance sensor is non-zero when driving with high frame-rate."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        game.game_config.fps = 100  # Put greater FPS to test the extremes
        
        # Update player and target position
        game.target = Vec2d(10, 1)
        game.set_player_init_pos(Vec2d(1, 1))
        game.set_player_init_angle(0)
        
        # Update the sensors used to only include the delta-distance sensor
        game.player.sensors = dict()
        game.player.add_delta_distance_sensor()
        game.player.add_distance_sensor()  # Last sensor must always be the distance sensor
        game.player.active_sensors = set(game.player.sensors.keys())
        
        # Drive forward for 10 simulated seconds
        start = True
        for _ in range(10 * game.game_config.fps):
            obs = game.step(l=1, r=1)
            if start:  # Cold start, reading of 0
                self.assertAlmostEqual(obs[D_SENSOR_LIST][0], 0.0, delta=EPSILON_DISTANCE)
                start = False
            else:
                self.assertGreater(obs[D_SENSOR_LIST][0], 0.0)  # Must be strictly greater than 0


class ProximitySensorTest(unittest.TestCase):
    """Test the proximity sensor."""
    
    def test_no_walls(self):
        """> Test proximity sensors with empty readings (i.e. no walls in proximity)."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        
        # Add walls to maze that are far enough from agent
        a = Vec2d(1, 1)
        b = Vec2d(1, 7)
        c = Vec2d(7, 7)
        d = Vec2d(7, 1)
        game.walls.update({Line2d(a, b), Line2d(b, c), Line2d(c, d), Line2d(d, a)})
        
        # Update player and target position
        game.set_player_init_pos(Vec2d(4, 4))  # Center of empty maze
        game.set_player_init_angle(0)
        
        # Only keep proximity sensors
        game.player.sensors = {k: v for k, v in game.player.sensors.items() if type(v) == ProximitySensor}
        game.player.active_sensors = set(game.player.sensors.keys())
        
        # Ask for the proximity-measures
        sensor_values = game.player.get_sensor_readings()
        for s in sensor_values:
            self.assertAlmostEqual(s, float(game.bot_config.ray_distance), delta=EPSILON_DISTANCE)
    
    def test_left_wall(self):
        """> Test when only wall on the agent's left."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        
        # Add walls to maze that are far enough from agent
        a = Vec2d(0, 5)
        b = Vec2d(5, 5)
        game.walls.add(Line2d(a, b))
        
        # Update player and target position
        game.set_player_init_pos(Vec2d(4, 4))  # Center of empty maze
        game.set_player_init_angle(0)  # Looking to the right
        
        # Reset the sensors with only one sensor to its left and one to its front
        game.player.sensors = dict()
        game.player.add_proximity_sensor(angle=np.pi / 2)  # To the agent's left
        game.player.add_proximity_sensor(angle=0)
        game.player.active_sensors = set(game.player.sensors.keys())
        
        # Ask for the proximity-measures
        sensor_values = game.player.get_sensor_readings()
        self.assertAlmostEqual(sensor_values[0], (1 - game.player.radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensor_values[1], game.bot_config.ray_distance, delta=EPSILON_DISTANCE)
    
    def test_cubed(self):
        """> Test proximity sensors when fully surrounded by walls."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        
        # Add walls to maze (cube around player)
        a = Vec2d(4, 5)
        b = Vec2d(5, 5)
        c = Vec2d(5, 3)
        d = Vec2d(4, 3)
        game.walls.update({Line2d(a, b), Line2d(b, c), Line2d(c, d)})
        
        # Update player position and angle
        game.set_player_init_pos(Vec2d(4, 4))
        game.set_player_init_angle(0)
        
        # Update sensors
        game.player.sensors = dict()  # Remove all previous set sensors
        game.player.add_proximity_sensor(np.pi / 2)  # 90° left (pointing upwards)
        game.player.add_proximity_sensor(np.pi / 4)  # 45° left
        game.player.add_proximity_sensor(0)  # 0°
        game.player.add_proximity_sensor(-np.pi / 4)  # 45° right
        game.player.add_proximity_sensor(-np.pi / 2)  # 90° right
        game.player.add_proximity_sensor(np.pi)  # 180° in the back
        game.player.active_sensors = set(game.player.sensors.keys())
        
        sensors = game.player.get_sensor_readings()
        self.assertAlmostEqual(sensors[0], 1.0 - float(game.bot_config.radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[1], np.sqrt(2) - float(game.bot_config.radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[2], 1 - float(game.bot_config.radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[3], np.sqrt(2) - float(game.bot_config.radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[4], 1 - float(game.bot_config.radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[5], game.bot_config.ray_distance, delta=EPSILON_DISTANCE)
    
    def test_force(self):
        """> Test proximity sensors when bot is grinding against a wall."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        
        # Add walls to maze
        b = Vec2d(5, 5)
        c = Vec2d(5, 3)
        game.walls.add(Line2d(b, c))
        
        # Update player position and angle
        game.set_player_init_pos(Vec2d(4, 4))
        game.set_player_init_angle(0)
        
        # Update sensors
        game.player.sensors = dict()
        game.player.add_proximity_sensor(0)  # 0°
        game.player.add_distance_sensor()  # Last sensor must always be the distance sensor
        game.player.active_sensors = set(game.player.sensors.keys())
        
        for _ in range(100): game.step(l=1, r=1)
        
        # Flat facing the wall, so upper sensor must always (approximately) equal zero
        for _ in range(50):
            sensor_values = game.player.get_sensor_readings()
            self.assertAlmostEqual(sensor_values[0], 0, delta=EPSILON_DISTANCE_L)


def get_game():
    cfg = Config()
    cfg.bot.ray_distance = RAY_DISTANCE
    return Game(game_id=0, config=cfg, silent=True, save_path="tests/games_db/", overwrite=True, noise=False)


def main():
    # Test angular sensors
    ast = AngularSensorTest()
    ast.test_front()
    ast.test_left_angle()
    
    # Test distance sensor
    dst = DistanceSensorTest()
    dst.test_front()
    dst.test_left_angle()
    
    # Test delta distance sensor
    delta_dst = DeltaDistanceSensorTest()
    delta_dst.test_front()
    delta_dst.test_equal_side()
    delta_dst.test_none_zero_drive()
    
    # Test proximity sensors
    pst = ProximitySensorTest()
    pst.test_no_walls()
    pst.test_cubed()
    pst.test_force()


if __name__ == '__main__':
    unittest.main()
