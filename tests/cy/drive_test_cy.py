import unittest
from random import random

import numpy as np

from environment.entities.cy.game_cy import GameCy
from utils.cy.line2d_cy import Line2dCy
from utils.cy.vec2d_cy import Vec2dCy

EPSILON = 0.05  # 5 centimeter offset allowed


class GameWallCollisionCy(unittest.TestCase):
    """
    Test the collision mechanism of the game.
    """
    
    def test_wall_force(self, save_path="../games_db/"):
        # Create empty game
        game = GameCy(save_path=save_path, silent=True, overwrite=True)
        
        # Drive forward for 100 seconds
        for _ in range(50 * game.fps):
            game.step(l=1, r=1)
        
        # Check if robot in fixed position
        self.assertAlmostEqual(game.player.pos.x, game.x_axis - 0.5, delta=EPSILON)
        self.assertAlmostEqual(game.player.pos.y, game.y_axis - game.bot_radius, delta=EPSILON)
    
    def test_wall_force_reverse(self, save_path="../games_db/"):
        # Create empty game
        game = GameCy(save_path=save_path, silent=True, overwrite=True)
        
        # Drive forward for 100 seconds
        for _ in range(10 * game.fps):
            game.step(l=-1, r=-1)
        
        # Check if robot in fixed position
        self.assertAlmostEqual(game.player.pos.x, game.x_axis - 0.5, delta=EPSILON)
        self.assertAlmostEqual(game.player.pos.y, game.bot_radius, delta=EPSILON)


class GameDriveCy(unittest.TestCase):
    """
    Test the robot's drive mechanics.
    """
    
    def test_360(self, save_path="../games_db/"):
        # Create empty game
        game = GameCy(save_path=save_path, silent=True, overwrite=True)
        
        # Keep spinning around
        for _ in range(10 * game.fps):
            game.step(l=-1, r=1)
        
        # Check if robot in fixed position
        self.assertAlmostEqual(game.player.pos.x, game.x_axis - 0.5, delta=EPSILON)
        self.assertAlmostEqual(game.player.pos.y, 0.5, delta=EPSILON)
    
    def test_remain_in_box(self, save_path="../games_db/"):
        """
        Set drone in small box in the middle of the game to check if it stays here in.
        """
        game = GameCy(save_path=save_path, silent=True, overwrite=True)
        
        # Create inner box
        a, b, c, d = Vec2dCy(4, 5), Vec2dCy(5, 5), Vec2dCy(5, 4), Vec2dCy(4, 4)
        game.walls += [Line2dCy(a, b), Line2dCy(b, c), Line2dCy(c, d), Line2dCy(d, a)]
        
        # Do 10 different initialized tests
        for _ in range(10):
            # Set player init positions
            game.set_player_pos(Vec2dCy(4.5, 4.5))
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
    # Test wall collision
    gwc = GameWallCollisionCy()
    gwc.test_wall_force(save_path="tests/games_db/")
    gwc.test_wall_force_reverse(save_path="tests/games_db/")
    
    # Test driving mechanics
    gd = GameDriveCy()
    gd.test_360(save_path="tests/games_db/")
    gd.test_remain_in_box(save_path="tests/games_db/")


if __name__ == '__main__':
    unittest.main()
