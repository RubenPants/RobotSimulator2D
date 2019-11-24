"""
Test of all the sensors.
"""

import unittest

from environment.cy_entities.god_class_cy import GameCy, Line2dCy, Vec2dCy
from utils.config import *

EPSILON_ANGLE = 0.0001  # 0.0001 radian offset allowed (~0.02 degrees)
EPSILON_DISTANCE = 0.001  # 1 millimeter offset allowed


class AngularSensorTestCy(unittest.TestCase):
    """
    Test the angular sensor.
    """
    
    def test_front(self, rel_path=""):
        game = GameCy(game_id=0, rel_path=rel_path, silent=True, overwrite=True, noise=False)
        
        # Update player and target position
        game.target = Vec2dCy(1, 0)
        game.set_player_pos(Vec2dCy(0, 0))
        game.set_player_angle(0)
        
        sensors = game.player.get_sensor_reading_angle()
        self.assertEqual(len(sensors), 2)
        for s in sensors.values():
            s = min(s, abs(s - 2 * np.pi))
            self.assertAlmostEqual(s, 0.0, delta=EPSILON_ANGLE)
    
    def test_left_angle(self, rel_path=""):
        game = GameCy(game_id=0, rel_path=rel_path, silent=True, overwrite=True, noise=False)
        
        # Update player and target position
        game.target = Vec2dCy(1, 1)
        game.set_player_pos(Vec2dCy(0, 0))
        game.set_player_angle(0)
        
        sensors = game.player.get_sensor_reading_angle()
        self.assertEqual(len(sensors), 2)
        self.assertAlmostEqual(sensors[0], 7 * np.pi / 4, delta=EPSILON_ANGLE)  # Clockwise
        self.assertAlmostEqual(sensors[1], np.pi / 4, delta=EPSILON_ANGLE)  # Anti-clockwise


class DistanceSensorTestCy(unittest.TestCase):
    """
    Test the distance sensor.
    """
    
    def test_front(self, rel_path=""):
        game = GameCy(game_id=0, rel_path=rel_path, silent=True, overwrite=True, noise=False)
        
        # Update player and target position
        game.target = Vec2dCy(1, 0)
        game.set_player_pos(Vec2dCy(0, 0))
        game.set_player_angle(0)
        
        self.assertAlmostEqual(game.player.get_sensor_reading_distance(), 1.0, delta=EPSILON_DISTANCE)
    
    def test_left_angle(self, rel_path=""):
        game = GameCy(game_id=0, rel_path=rel_path, silent=True, overwrite=True, noise=False)
        
        # Update player and target position
        game.target = Vec2dCy(1, 1)
        game.set_player_pos(Vec2dCy(0, 0))
        game.set_player_angle(0)
        
        self.assertAlmostEqual(game.player.get_sensor_reading_distance(), np.sqrt(2), delta=EPSILON_DISTANCE)


class ProximitySensorTestCy(unittest.TestCase):
    """
    Test the proximity sensor.
    """
    
    def test_no_walls(self, rel_path=""):
        game = GameCy(game_id=0, rel_path=rel_path, silent=True, overwrite=True, noise=False)
        
        # Add walls to maze that are far enough from agent
        a = Vec2dCy(2, 2)
        b = Vec2dCy(2, 6)
        c = Vec2dCy(6, 6)
        d = Vec2dCy(6, 2)
        game.walls += [Line2dCy(a, b), Line2dCy(b, c), Line2dCy(c, d), Line2dCy(d, a)]
        
        # Update player and target position
        game.set_player_pos(Vec2dCy(4, 4))  # Center of empty maze
        game.set_player_angle(0)
        
        sensors = game.player.get_sensor_reading_proximity()
        for s in sensors.values():
            self.assertAlmostEqual(s, SENSOR_RAY_DISTANCE, delta=EPSILON_DISTANCE)
    
    def test_cubed(self, rel_path=""):
        game = GameCy(game_id=0, rel_path=rel_path, silent=True, overwrite=True, noise=False)
        
        # Add walls to maze
        a = Vec2dCy(4, 5)
        b = Vec2dCy(5, 5)
        c = Vec2dCy(5, 3)
        d = Vec2dCy(4, 3)
        game.walls += [Line2dCy(a, b), Line2dCy(b, c), Line2dCy(c, d)]
        
        # Update player position and angle
        game.set_player_pos(Vec2dCy(4, 4))
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
    rel_path = "environment/cy_entities/"
    
    # Test angular sensors
    ast = AngularSensorTestCy()
    ast.test_front(rel_path=rel_path)
    ast.test_left_angle(rel_path=rel_path)
    
    # Test distance sensor
    dst = DistanceSensorTestCy()
    dst.test_front(rel_path=rel_path)
    dst.test_left_angle(rel_path=rel_path)
    
    # Test proximity sensors
    pst = ProximitySensorTestCy()
    pst.test_no_walls(rel_path=rel_path)
    pst.test_cubed(rel_path=rel_path)


if __name__ == '__main__':
    unittest.main()
