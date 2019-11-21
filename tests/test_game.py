"""
Tests concerning the game mechanics.
"""

import unittest

from environment.entities.game import Game
from utils.config import *


class GameWallCollision(unittest.TestCase):
    """
    Test the collision mechanism of the game.
    """
    
    def test_wall_force(self):
        # Create empty game
        game = Game(game_id=0, rel_path="")
        
        # Drive forward for 100 seconds
        for _ in range(50 * FPS):
            game.step(l=1, r=1)
        
        # Check if robot in fixed position
        eps = 0.05  # Meter offset allowed
        self.assertAlmostEqual(game.player.pos.x, AXIS_X - 0.5, delta=eps)
        self.assertAlmostEqual(game.player.pos.y, AXIS_Y, delta=eps)
    
    def test_wall_force_reverse(self):
        # Create empty game
        game = Game(game_id=0, rel_path="")
        
        # Drive forward for 100 seconds
        for _ in range(10 * FPS):
            game.step(l=-1, r=-1)
        
        # Check if robot in fixed position
        eps = 0.05  # Meter offset allowed
        self.assertAlmostEqual(game.player.pos.x, AXIS_X - 0.5, delta=eps)
        self.assertAlmostEqual(game.player.pos.y, 0, delta=eps)


class GameDrive(unittest.TestCase):
    """
    Test the robot's drive mechanics.
    """
    
    def test_360(self):
        # Create empty game
        game = Game(game_id=0, rel_path="")
        
        # Keep spinning around
        for _ in range(10 * FPS):
            game.step(l=-1, r=1)
        
        # Check if robot in fixed position
        eps = 0.05  # Meter offset allowed
        self.assertAlmostEqual(game.player.pos.x, AXIS_X - 0.5, delta=eps)
        self.assertAlmostEqual(game.player.pos.y, 0.5, delta=eps)


if __name__ == '__main__':
    unittest.main()
