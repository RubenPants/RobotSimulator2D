"""
Test of all the sensors.
"""

import unittest

from environment.entities.game import Game
from utils.config import *
from utils.line2d import Line2d
from utils.vec2d import Vec2d

EPSILON_ANGLE = 0.0001  # 0.0001 radian offset allowed (~0.02 degrees)
EPSILON_DISTANCE = 0.001  # 1 millimeter offset allowed


class AngularSensorTest(unittest.TestCase):
    """
    Test the angular sensor.
    """
    
    def test_front(self, rel_path=""):
        game = Game(game_id=0, rel_path=rel_path, silent=True, overwrite=True, noise=False)
        
        # Update player and target position
        game.target = Vec2d(1, 0)
        game.set_player_pos(Vec2d(0, 0))
        game.set_player_angle(0)
        
        sensors = game.player.get_sensor_reading_angle()
        self.assertEqual(len(sensors), 2)
        for s in sensors.values():
            s = min(s, abs(s - 2 * np.pi))
            self.assertAlmostEqual(s, 0.0, delta=EPSILON_ANGLE)
    
    def test_left_angle(self, rel_path=""):
        game = Game(game_id=0, rel_path=rel_path, silent=True, overwrite=True, noise=False)
        
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
    
    def test_front(self, rel_path=""):
        game = Game(game_id=0, rel_path=rel_path, silent=True, overwrite=True, noise=False)
        
        # Update player and target position
        game.target = Vec2d(1, 0)
        game.set_player_pos(Vec2d(0, 0))
        game.set_player_angle(0)
        
        self.assertAlmostEqual(game.player.get_sensor_reading_distance(), 1.0, delta=EPSILON_DISTANCE)
    
    def test_left_angle(self, rel_path=""):
        game = Game(game_id=0, rel_path=rel_path, silent=True, overwrite=True, noise=False)
        
        # Update player and target position
        game.target = Vec2d(1, 1)
        game.set_player_pos(Vec2d(0, 0))
        game.set_player_angle(0)
        
        self.assertAlmostEqual(game.player.get_sensor_reading_distance(), np.sqrt(2), delta=EPSILON_DISTANCE)


class ProximitySensorTest(unittest.TestCase):
    """
    Test the proximity sensor.
    """
    
    def test_no_walls(self, rel_path=""):
        game = Game(game_id=0, rel_path=rel_path, silent=True, overwrite=True, noise=False)
        
        # Update player and target position
        game.set_player_pos(Vec2d(AXIS_X / 2, AXIS_Y / 2))  # Center of empty maze
        game.set_player_angle(0)
        
        sensors = game.player.get_sensor_reading_proximity()
        for s in sensors.values():
            self.assertAlmostEqual(s, SENSOR_RAY_DISTANCE, delta=EPSILON_DISTANCE)
    
    def test_cubed(self, rel_path=""):
        game = Game(game_id=0, rel_path=rel_path, silent=True, overwrite=True, noise=False)
        
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
        game.player.proximity_sensors = set()
        game.player.add_proximity_sensor(np.pi / 2)  # 90° left (pointing upwards)
        game.player.add_proximity_sensor(np.pi / 4)  # 45° left
        game.player.add_proximity_sensor(0)  # 0°
        game.player.add_proximity_sensor(-np.pi / 4)  # 45° right
        game.player.add_proximity_sensor(-np.pi / 2)  # 90° right
        game.player.add_proximity_sensor(np.pi)  # 180° in the back
        
        sensors = game.player.get_sensor_reading_proximity()
        self.assertAlmostEqual(sensors[0], 1.0 - BOT_RADIUS, delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[1], np.sqrt(2) - BOT_RADIUS, delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[2], 1 - BOT_RADIUS, delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[3], np.sqrt(2) - BOT_RADIUS, delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[4], 1 - BOT_RADIUS, delta=EPSILON_DISTANCE)
        self.assertAlmostEqual(sensors[5], SENSOR_RAY_DISTANCE, delta=EPSILON_DISTANCE)


def main():
    rel_path = "tests/"
    
    # Test angular sensors
    ast = AngularSensorTest()
    ast.test_front(rel_path=rel_path)
    ast.test_left_angle(rel_path=rel_path)
    
    # Test distance sensor
    dst = DistanceSensorTest()
    dst.test_front(rel_path=rel_path)
    dst.test_left_angle(rel_path=rel_path)
    
    # Test proximity sensors
    pst = ProximitySensorTest()
    pst.test_no_walls(rel_path=rel_path)
    pst.test_cubed(rel_path=rel_path)


if __name__ == '__main__':
    unittest.main()
