"""
Test of all the sensors.
"""
import os
import unittest

import numpy as np

from environment.entities.game import Game
from utils.line2d import Line2d
from utils.vec2d import Vec2d

# Parameters
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
        game = Game(silent=True, save_path="tests/games_db/", overwrite=True, noise=False)
        
        # Update player and target position
        game.target = Vec2d(1, 0)
        game.set_player_pos(Vec2d(0, 0))
        game.set_player_angle(0)
        
        sensors = game.player.get_sensor_reading_angle()
        self.assertEqual(len(sensors), 2)
        for s in sensors.values():
            s = min(s, abs(s - 2 * np.pi))
            self.assertAlmostEqual(s, 0.0, delta=EPSILON_ANGLE)
    
    def test_left_angle(self):
        """> Test the angular sensors when target on the left."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = Game(silent=True, save_path="tests/games_db/", overwrite=True, noise=False)
        
        # Update player and target position
        game.target = Vec2d(1, 1)
        game.set_player_pos(Vec2d(0, 0))
        game.set_player_angle(0)
        
        sensors = game.player.get_sensor_reading_angle()
        self.assertEqual(len(sensors), 2)
        self.assertAlmostEqual(sensors[0], 7 * np.pi / 4, delta=EPSILON_ANGLE)  # Clockwise
        self.assertAlmostEqual(sensors[1], np.pi / 4, delta=EPSILON_ANGLE)  # Anti-clockwise


class DistanceSensorTest(unittest.TestCase):
    """
    Test the distance sensor.
    """
    
    def test_front(self):
        """> Test the distance sensor when target straight in the front."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = Game(silent=True, save_path="tests/games_db/", overwrite=True, noise=False)
        
        # Update player and target position
        game.target = Vec2d(1, 0)
        game.set_player_pos(Vec2d(0, 0))
        game.set_player_angle(0)
        
        self.assertAlmostEqual(game.player.get_sensor_reading_distance(), 1.0, delta=EPSILON_DISTANCE)
    
    def test_left_angle(self):
        """> Test distance sensor when target under an angle (towards the left)."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = Game(silent=True, save_path="tests/games_db/", overwrite=True, noise=False)
        
        # Update player and target position
        game.target = Vec2d(1, 1)
        game.set_player_pos(Vec2d(0, 0))
        game.set_player_angle(0)
        
        self.assertAlmostEqual(game.player.get_sensor_reading_distance(), np.sqrt(2), delta=EPSILON_DISTANCE)


class ProximitySensorTest(unittest.TestCase):
    """Test the proximity sensor."""
    
    def test_no_walls(self):
        """> Test proximity sensors with empty readings (i.e. no walls in proximity)."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = Game(silent=True, save_path="tests/games_db/", overwrite=True, noise=False)
        
        # Add walls to maze that are far enough from agent
        a = Vec2d(2, 2)
        b = Vec2d(2, 6)
        c = Vec2d(6, 6)
        d = Vec2d(6, 2)
        game.walls += [Line2d(a, b), Line2d(b, c), Line2d(c, d), Line2d(d, a)]
        
        # Update player and target position
        game.set_player_pos(Vec2d(4, 4))  # Center of empty maze
        game.set_player_angle(0)
        
        sensors = game.player.get_sensor_reading_proximity()
        for s in sensors.values():
            self.assertAlmostEqual(s, float(game.ray_distance), delta=EPSILON_DISTANCE)
    
    def test_cubed(self):
        """> Test proximity sensors when fully surrounded by walls."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = Game(silent=True, save_path="tests/games_db/", overwrite=True, noise=False)
        
        # Add walls to maze
        a = Vec2d(4, 5)
        b = Vec2d(5, 5)
        c = Vec2d(5, 3)
        d = Vec2d(4, 3)
        game.walls += [Line2d(a, b), Line2d(b, c), Line2d(c, d)]
        
        # Update player position and angle
        game.set_player_pos(Vec2d(4, 4))
        game.set_player_angle(0)
        
        # Update sensors
        game.player.proximity_sensors = list()
        game.player.add_proximity_sensor(np.pi / 2)  # 90° left (pointing upwards)
        game.player.add_proximity_sensor(np.pi / 4)  # 45° left
        game.player.add_proximity_sensor(0)  # 0°
        game.player.add_proximity_sensor(-np.pi / 4)  # 45° right
        game.player.add_proximity_sensor(-np.pi / 2)  # 90° right
        game.player.add_proximity_sensor(np.pi)  # 180° in the back
        
        sensors = game.player.get_sensor_reading_proximity()
        self.assertAlmostEqual(sensors[0], 1.0 - float(game.bot_radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[1], np.sqrt(2) - float(game.bot_radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[2], 1 - float(game.bot_radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[3], np.sqrt(2) - float(game.bot_radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[4], 1 - float(game.bot_radius), delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[5], game.ray_distance, delta=EPSILON_DISTANCE)
    
    def test_force(self):
        """> Test proximity sensors when bot is grinding against a wall."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create empty game
        game = Game(silent=True, save_path="tests/games_db/", overwrite=True, noise=False)
        
        # Add walls to maze
        b = Vec2d(5, 5)
        c = Vec2d(5, 3)
        game.walls += [Line2d(b, c)]
        
        # Update player position and angle
        game.set_player_pos(Vec2d(4, 4))
        game.set_player_angle(0)
        
        # Update sensors
        game.player.proximity_sensors = list()
        game.player.add_proximity_sensor(0)  # 0°
        
        for _ in range(100): game.step(l=1, r=1)
        
        # Flat facing the wall, so upper sensor must always (approximately) equal zero
        for _ in range(50):
            sensors = game.player.get_sensor_reading_proximity()
            self.assertAlmostEqual(sensors[0], 0, delta=EPSILON_DISTANCE_L)


def main():
    # Test angular sensors
    ast = AngularSensorTest()
    ast.test_front()
    ast.test_left_angle()
    
    # Test distance sensor
    dst = DistanceSensorTest()
    dst.test_front()
    dst.test_left_angle()
    
    # Test proximity sensors
    pst = ProximitySensorTest()
    pst.test_no_walls()
    pst.test_cubed()
    pst.test_force()


if __name__ == '__main__':
    unittest.main()
