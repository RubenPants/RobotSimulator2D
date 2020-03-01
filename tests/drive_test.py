"""
Tests concerning the game mechanics.
"""
import unittest
from random import random

import numpy as np

from environment.entities.game import Game
from utils.line2d import Line2d
from utils.vec2d import Vec2d

EPSILON = 0.05  # 5 centimeter offset allowed


class GameWallCollision(unittest.TestCase):
    """
    Test the collision mechanism of the game.
    """
    
    def test_wall_force(self, save_path="games_db/"):
        # Create empty game
        game = Game(silent=True, save_path=save_path, overwrite=True)
        
        # Drive forward for 100 seconds
        for _ in range(50 * game.fps):
            game.step(l=1, r=1)
        
        # Check if robot in fixed position
        self.assertAlmostEqual(game.player.pos.x, int(game.x_axis) - 0.5, delta=EPSILON)
        self.assertAlmostEqual(game.player.pos.y,
                               int(game.y_axis) - float(game.bot_radius),
                               delta=EPSILON)
    
    def test_wall_force_reverse(self, save_path="games_db/"):
        # Create empty game
        game = Game(silent=True, save_path=save_path, overwrite=True)
        
        # Drive forward for 100 seconds
        for _ in range(10 * game.fps):
            game.step(l=-1, r=-1)
        
        # Check if robot in fixed position
        self.assertAlmostEqual(game.player.pos.x, int(game.x_axis) - 0.5, delta=EPSILON)
        self.assertAlmostEqual(game.player.pos.y, float(game.bot_radius), delta=EPSILON)
        self.assertAlmostEqual(game.player.pos.y, float(game.bot_radius), delta=EPSILON)


class GameDrive(unittest.TestCase):
    """
    Test the robot's drive mechanics.
    """
    
    def test_360(self, save_path="games_db/"):
        """ Let bot spin 360s and check if position has changed. """
        # Create empty game
        game = Game(silent=True, save_path=save_path, overwrite=True)
        
        # Keep spinning around
        for _ in range(10 * game.fps):
            game.step(l=-1, r=1)
        
        # Check if robot in fixed position
        self.assertAlmostEqual(game.player.pos.x, int(game.x_axis) - 0.5, delta=EPSILON)
        self.assertAlmostEqual(game.player.pos.y, 0.5, delta=EPSILON)
    
    def test_remain_in_box(self, save_path="games_db/"):
        """ Set drone in small box in the middle of the game to check if it stays here in. """
        # Create empty game
        game = Game(silent=True, save_path=save_path, overwrite=True)
        
        # Create inner box
        a, b, c, d = Vec2d(4, 5), Vec2d(5, 5), Vec2d(5, 4), Vec2d(4, 4)
        game.walls += [Line2d(a, b), Line2d(b, c), Line2d(c, d), Line2d(d, a)]
        
        # Do 10 different initialized tests
        for _ in range(10):
            # Set player init positions
            game.set_player_pos(Vec2d(4.5, 4.5))
            game.player.angle = random() * np.pi
            
            # Run for one twenty seconds
            for _ in range(20 * game.fps):
                l = random() * 1.5 - 0.5  # [-0.5..1]
                r = random() * 1.5 - 0.5  # [-0.5..1]
                game.step(l=l, r=r)
            
            # Check if bot still in box
            self.assertTrue(4 <= game.player.pos.x <= 5)
            self.assertTrue(4 <= game.player.pos.y <= 5)


def main():
    # Counters
    success, fail = 0, 0
    
    # Test wall collision
    gwc = GameWallCollision()
    try:
        gwc.test_wall_force(save_path="tests/games_db/")
        success += 1
    except AssertionError:
        fail += 1
    try:
        gwc.test_wall_force_reverse(save_path="tests/games_db/")
        success += 1
    except AssertionError:
        fail += 1
    
    # Test driving mechanics
    gd = GameDrive()
    try:
        gd.test_360(save_path="tests/games_db/")
        success += 1
    except AssertionError:
        fail += 1
    try:
        gd.test_remain_in_box(save_path="tests/games_db/")
        success += 1
    except AssertionError:
        fail += 1
    
    return success, fail


if __name__ == '__main__':
    unittest.main()
