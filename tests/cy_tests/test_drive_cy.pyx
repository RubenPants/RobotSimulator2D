import unittest
from random import random

from ..game_cy cimport GameCy
from ..line2d_cy cimport Line2dCy
from ..vec2d_cy cimport Vec2dCy


EPSILON = 0.05  # 5 centimeter offset allowed

cdef class GameWallCollisionCy:
    """
    Test the collision mechanism of the game.
    """
    
    cpdef test_wall_force(self, str rel_path=""):
        # Create empty game
        cdef GameCy game
        game = GameCy(rel_path=rel_path, silent=True, overwrite=True)
        
        # Drive forward for 100 seconds
        for _ in range(50 * FPS):
            game.step(l=1, r=1)
        
        # Check if robot in fixed position
        assert(AXIS_X - 0.5 - EPSILON <= game.player.pos.x <= AXIS_X - 0.5 + EPSILON)
        assert(AXIS_Y - BOT_RADIUS - EPSILON < game.player.pos.y < AXIS_Y - BOT_RADIUS + EPSILON)
    
    cpdef test_wall_force_reverse(self, str rel_path=""):
        # Create empty game
        cdef GameCy game
        game = GameCy(rel_path=rel_path, silent=True, overwrite=True)
        
        # Drive forward for 100 seconds
        for _ in range(10 * FPS):
            game.step(l=-1, r=-1)
        
        # Check if robot in fixed position
        assert(AXIS_X - 0.5 - EPSILON < game.player.pos.x < AXIS_X - 0.5 + EPSILON)
        assert(BOT_RADIUS - EPSILON < game.player.pos.y < BOT_RADIUS + EPSILON)

cdef class GameDriveCy:
    """
    Test the robot's drive mechanics.
    """
    
    cpdef test_360(self, str rel_path=""):
        # Create empty game
        cdef GameCy game
        game = GameCy(rel_path=rel_path, silent=True, overwrite=True)
        
        # Keep spinning around
        for _ in range(10 * FPS):
            game.step(l=-1, r=1)
        
        # Check if robot in fixed position
        assert(AXIS_X - 0.5 - EPSILON < game.player.pos.x < AXIS_X - 0.5 + EPSILON)
        assert(0.5 - EPSILON < game.player.pos.y < 0.5 + EPSILON)
    
    cpdef test_remain_in_box(self, str rel_path=""):
        """
        Set drone in small box in the middle of the game to check if it stays here in.
        """
        cdef Vec2dCy a, b, c, d
        cdef float l, r
        cdef GameCy game
        game = GameCy(rel_path=rel_path, silent=True, overwrite=True)
        
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
            assert(4 <= game.player.pos.x <= 5)
            assert(4 <= game.player.pos.y <= 5)

cpdef main():
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
