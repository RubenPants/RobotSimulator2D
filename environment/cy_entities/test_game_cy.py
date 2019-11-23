import unittest
from random import random

from environment.cy_entities.god_class_cy import GameCy, Line2dCy, Vec2dCy
from utils.config import *


class GameWallCollisionCy(unittest.TestCase):
    """
    Test the collision mechanism of the game.
    """
    
    def test_wall_force(self, rel_path=""):
        # Create empty game
        game = GameCy(game_id=0, rel_path=rel_path, silent=True)
        
        # Drive forward for 100 seconds
        for _ in range(50 * FPS):
            game.step(l=1, r=1)
        
        # Check if robot in fixed position
        eps = 0.05  # Meter offset allowed
        self.assertAlmostEqual(game.player.pos.x, AXIS_X - 0.5, delta=eps)
        self.assertAlmostEqual(game.player.pos.y, AXIS_Y, delta=eps)
    
    def test_wall_force_reverse(self, rel_path=""):
        # Create empty game
        game = GameCy(game_id=0, rel_path=rel_path, silent=True)
        
        # Drive forward for 100 seconds
        for _ in range(10 * FPS):
            game.step(l=-1, r=-1)
        
        # Check if robot in fixed position
        eps = 0.05  # Meter offset allowed
        self.assertAlmostEqual(game.player.pos.x, AXIS_X - 0.5, delta=eps)
        self.assertAlmostEqual(game.player.pos.y, 0, delta=eps)


class GameDriveCy(unittest.TestCase):
    """
    Test the robot's drive mechanics.
    """
    
    def test_360(self, rel_path=""):
        # Create empty game
        game = GameCy(game_id=0, rel_path=rel_path, silent=True)
        
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
        game = GameCy(game_id=1, rel_path=rel_path, overwrite=True, silent=True)
        
        # Create inner box
        a, b, c, d = Vec2dCy(4, 5), Vec2dCy(5, 5), Vec2dCy(5, 4), Vec2dCy(4, 4)
        game.walls += [Line2dCy(a, b), Line2dCy(b, c), Line2dCy(c, d), Line2dCy(d, a)]
        
        # Do 10 different initialized tests
        for _ in range(10):
            # Set player init positions
            game.set_player_pos(Vec2dCy(4.5, 4.5))
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
    rel_path = "environment/cy_entities/"
    
    # Test wall collision
    gwc = GameWallCollisionCy()
    gwc.test_wall_force(rel_path=rel_path)
    gwc.test_wall_force_reverse(rel_path=rel_path)
    
    # Test driving mechanics
    gd = GameDriveCy()
    gd.test_360(rel_path=rel_path)
    gd.test_remain_in_box(rel_path=rel_path)


if __name__ == '__main__':
    unittest.main()
