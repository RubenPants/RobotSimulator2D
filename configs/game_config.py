"""
game_config.py

Configuration file relating to the game.
"""
from configs.base_config import BaseConfig


class GameConfig(BaseConfig):
    """Game-specific configuration parameters."""
    
    __slots__ = {
        'batch', 'duration', 'max_game_id', 'max_eval_game_id', 'fps', 'p2m', 'target_reached', 'x_axis', 'y_axis',
    }
    
    def __init__(self):
        # Number of games on which a single genome is evaluated  [def=12]  TODO
        self.batch: int = 10
        # Number of seconds it takes for one game to complete  [def=60]  TODO
        self.duration: int = 60
        # Max ID of game (starting from 1)  [def=1000]
        self.max_game_id: int = 1000
        # Max ID of evaluation game (starting from max_id)  [def=1100]
        self.max_eval_game_id: int = 1100
        # Number of frames each second  [def=20]
        self.fps: int = 20
        # Pixel-to-meters: number of pixels that represent one meter  [def=50]
        self.p2m: int = 50
        # Target is reached when within this range, expressed in meters  [def=0.5]
        self.target_reached: float = 0.5
        # Number of meters the x-axis represents  [def=14]
        self.x_axis: int = 14
        # Number of meters the y-axis represents  [def=14]
        self.y_axis: int = 14
