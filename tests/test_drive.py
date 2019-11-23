"""
Tests concerning the game mechanics.
"""
import unittest
from random import random

from environment.entities.game import Game
from utils.config import *
from utils.line2d import Line2d
from utils.vec2d import Vec2d


class GameWallCollision(unittest.TestCase):
    """
    Test the collision mechanism of the game.
    """
    
    def test_wall_force(self, rel_path=""):
        # Create empty game
        game = Game(game_id=0, rel_path=rel_path, silent=True, overwrite=True)
        
        # Drive forward for 100 seconds
        for _ in range(50 * FPS):
            game.step(l=1, r=1)
        
        # Check if robot in fixed position
        eps = 0.05  # Meter offset allowed
        self.assertAlmostEqual(game.player.pos.x, AXIS_X - 0.5, delta=eps)
        self.assertAlmostEqual(game.player.pos.y, AXIS_Y - BOT_RADIUS, delta=eps)
    
    def test_wall_force_reverse(self, rel_path=""):
        # Create empty game
        game = Game(game_id=0, rel_path=rel_path, silent=True, overwrite=True)
        
        # Drive forward for 100 seconds
        for _ in range(10 * FPS):
            game.step(l=-1, r=-1)
        
        # Check if robot in fixed position
        eps = 0.05  # Meter offset allowed
        self.assertAlmostEqual(game.player.pos.x, AXIS_X - 0.5, delta=eps)
        self.assertAlmostEqual(game.player.pos.y, BOT_RADIUS, delta=eps)


class GameDrive(unittest.TestCase):
    """
    Test the robot's drive mechanics.
    """
    
    def test_360(self, rel_path=""):
        # Create empty game
        game = Game(game_id=0, rel_path=rel_path, silent=True, overwrite=True)
        
        # Keep spinning around
        for _ in range(10 * FPS):
            game.step(l=-1, r=1)
        
        # Check if robot in fixed position
        eps = 0.05  # Meter offset allowed
        self.assertAlmostEqual(game.player.pos.x, AXIS_X - 0.5, delta=eps)
        self.assertAlmostEqual(game.player.pos.y, 0.5, delta=eps)
    
    def test_remain_in_box(self, rel_path=""):
        """
        Set drone in small box in the middle of the game to check if it stays here in.
        """
        game = Game(game_id=1, rel_path=rel_path, silent=True, overwrite=True)
        
        # Create inner box
        a, b, c, d = Vec2d(4, 5), Vec2d(5, 5), Vec2d(5, 4), Vec2d(4, 4)
        game.walls += [Line2d(a, b), Line2d(b, c), Line2d(c, d), Line2d(d, a)]
        
        # Do 10 different initialized tests
        for _ in range(10):
            # Set player init positions
            game.set_player_pos(Vec2d(4.5, 4.5))
            game.player.angle = random() * np.pi
            
            # Run for one twenty seconds
            for _ in range(20 * FPS):
                l = random() * 1.5 - 0.5  # [-0.5..1]
                r = random() * 1.5 - 0.5  # [-0.5..1]
                game.step(l=l, r=r)
            
            # Check if bot still in box
            self.assertTrue(4 <= game.player.pos.x <= 5)
            self.assertTrue(4 <= game.player.pos.y <= 5)


def main():
    rel_path = "tests/"
    
    # Test wall collision
    gwc = GameWallCollision()
    gwc.test_wall_force(rel_path=rel_path)
    gwc.test_wall_force_reverse(rel_path=rel_path)
    
    # Test driving mechanics
    gd = GameDrive()
    gd.test_360(rel_path=rel_path)
    gd.test_remain_in_box(rel_path=rel_path)


if __name__ == '__main__':
    unittest.main()
