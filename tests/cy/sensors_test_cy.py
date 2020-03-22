"""
sensor_test_cy.py

Test of all the sensors.
"""
import os
import unittest

import numpy as np

from config import GameConfig
from environment.entities.cy.game_cy import GameCy
from environment.entities.cy.sensors_cy import ProximitySensorCy
from utils.cy.line2d_cy import Line2dCy
from utils.cy.vec2d_cy import Vec2dCy

# Parameters
RAY_DISTANCE = 2  # Maximum range of the proximity-sensor
EPSILON_ANGLE = 0.0001  # 0.0001 radian offset allowed (~0.02 degrees)
EPSILON_DISTANCE = 0.001  # 1 millimeter offset allowed
EPSILON_DISTANCE_L = 0.1  # 10 centimeter offset allowed


class AngularSensorTestCy(unittest.TestCase):
    """Test the angular sensor."""
    
    def test_front(self):
        """> Test angular sensors when target straight in the front."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2dCy(1, 0)
        game.set_player_init_pos(Vec2dCy(0, 0))
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
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2dCy(1, 1)
        game.set_player_init_pos(Vec2dCy(0, 0))
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


class DistanceSensorTestCy(unittest.TestCase):
    """
    Test the distance sensor.
    """
    
    def test_front(self):
        """> Test the distance sensor when target straight in the front."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2dCy(1, 0)
        game.set_player_init_pos(Vec2dCy(0, 0))
        game.set_player_init_angle(0)
        
        # Ask for the distance
        game.get_observation()
        self.assertAlmostEqual(game.player.get_sensor_readings_distance(), 1.0, delta=EPSILON_DISTANCE)
    
    def test_left_angle(self):
        """> Test distance sensor when target under an angle (towards the left)."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Create empty game
        game = get_game()
        
        # Update player and target position
        game.target = Vec2dCy(1, 1)
        game.set_player_init_pos(Vec2dCy(0, 0))
        game.set_player_init_angle(0)
        
        # Ask for the distance
        game.get_observation()
        self.assertAlmostEqual(game.player.get_sensor_readings_distance(), np.sqrt(2), delta=EPSILON_DISTANCE)


class ProximitySensorTestCy(unittest.TestCase):
    """Test the proximity sensor."""
    
    def test_no_walls(self):
        """> Test proximity sensors with empty readings (i.e. no walls in proximity)."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Create empty game
        game = get_game()
        
        # Add walls to maze that are far enough from agent
        a = Vec2dCy(1, 1)
        b = Vec2dCy(1, 7)
        c = Vec2dCy(7, 7)
        d = Vec2dCy(7, 1)
        game.walls.update({Line2dCy(a, b), Line2dCy(b, c), Line2dCy(c, d), Line2dCy(d, a)})
        
        # Update player and target position
        game.set_player_init_pos(Vec2dCy(4, 4))  # Center of empty maze
        game.set_player_init_angle(0)
        
        # Only keep proximity sensors
        game.player.sensors = {k: v for k, v in game.player.sensors.items() if type(v) == ProximitySensorCy}
        game.player.active_sensors = set(game.player.sensors.keys())
        
        # Ask for the proximity-measures
        sensor_values = game.player.get_sensor_readings()
        for s in sensor_values:
            self.assertAlmostEqual(s, float(game.ray_distance), delta=EPSILON_DISTANCE)
    
    def test_left_wall(self):
        """> Test when only wall on the agent's left."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = get_game()
        
        # Add walls to maze that are far enough from agent
        a = Vec2dCy(0, 5)
        b = Vec2dCy(5, 5)
        game.walls.add(Line2dCy(a, b))
        
        # Update player and target position
        game.set_player_init_pos(Vec2dCy(4, 4))  # Center of empty maze
        game.set_player_init_angle(0)  # Looking to the right
        
        # Reset the sensors with only one sensor to its left and one to its front
        game.player.sensors = dict()
        game.player.add_proximity_sensor(angle=np.pi / 2)  # To the agent's left
        game.player.add_proximity_sensor(angle=0)
        game.player.active_sensors = set(game.player.sensors.keys())
        
        # Ask for the proximity-measures
        sensor_values = game.player.get_sensor_readings()
        self.assertAlmostEqual(sensor_values[0], (1 - game.player.radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensor_values[1], game.ray_distance, delta=EPSILON_DISTANCE)
    
    def test_cubed(self):
        """> Test proximity sensors when fully surrounded by walls."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Create empty game
        game = get_game()
        
        # Add walls to maze (cube around player)
        a = Vec2dCy(4, 5)
        b = Vec2dCy(5, 5)
        c = Vec2dCy(5, 3)
        d = Vec2dCy(4, 3)
        game.walls.update({Line2dCy(a, b), Line2dCy(b, c), Line2dCy(c, d)})
        
        # Update player position and angle
        game.set_player_init_pos(Vec2dCy(4, 4))
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
        self.assertAlmostEqual(sensors[0], 1.0 - float(game.bot_radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[1], np.sqrt(2) - float(game.bot_radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[2], 1 - float(game.bot_radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[3], np.sqrt(2) - float(game.bot_radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[4], 1 - float(game.bot_radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[5], game.ray_distance, delta=EPSILON_DISTANCE)
    
    def test_force(self):
        """> Test proximity sensors when bot is grinding against a wall."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('../..')
        
        # Create empty game
        game = get_game()
        
        # Add walls to maze
        b = Vec2dCy(5, 5)
        c = Vec2dCy(5, 3)
        game.walls.add(Line2dCy(b, c))
        
        # Update player position and angle
        game.set_player_init_pos(Vec2dCy(4, 4))
        game.set_player_init_angle(0)
        
        # Update sensors
        game.player.sensors = dict()
        game.player.add_proximity_sensor(0)  # 0°
        game.player.active_sensors = set(game.player.sensors.keys())
        
        for _ in range(100): game.step(l=1, r=1)
        
        # Flat facing the wall, so upper sensor must always (approximately) equal zero
        for _ in range(50):
            sensor_values = game.player.get_sensor_readings()
            self.assertAlmostEqual(sensor_values[0], 0, delta=EPSILON_DISTANCE_L)


def get_game():
    cfg = GameConfig()
    cfg.sensor_ray_distance = RAY_DISTANCE
    return GameCy(game_id=0, config=cfg, silent=True, save_path="tests/games_db/", overwrite=True, noise=False)


def main():
    # Test angular sensors
    ast = AngularSensorTestCy()
    ast.test_front()
    ast.test_left_angle()
    
    # Test distance sensor
    dst = DistanceSensorTestCy()
    dst.test_front()
    dst.test_left_angle()
    
    # Test proximity sensors
    pst = ProximitySensorTestCy()
    pst.test_no_walls()
    pst.test_cubed()
    pst.test_force()


if __name__ == '__main__':
    unittest.main()
