"""
visualizer.py

TODO
"""

import pyglet
import pymunk
from pymunk.pyglet_util import DrawOptions

from environment.entities.game import Game
from utils.config import *
from utils.dictionary import D_SENSOR_LIST


class Visualizer:
    """
    The visualizer will visualize the run of a single genome from the population in a game of choice. This is done by
    the use of pymunk.
    """
    
    def __init__(self,
                 query_net,
                 debug: bool = True,
                 rel_path: str = '',
                 speedup: float = 3):
        """
        The visualizer provides methods used to visualize the performance of a single genome.
        
        :param query_net: Method used to query the network
        :param debug: Generates prints (CLI) during visualization
        :param rel_path: Relative path pointing to the 'environment/' folder
        :param speedup: Specifies the relative speedup the virtual environment faces towards the real world
        """
        # Set relative path
        self.rel_path = '{rp}{x}'.format(rp=rel_path, x='/' if (rel_path and rel_path[-1] not in ['/', '\\']) else '')
        
        # Visualizer specific parameters
        self.speedup = speedup
        self.state = None
        
        # Network specific parameters
        self.query_net = query_net
        
        # Debug options
        self.debug = debug
    
    def visualize(self, network, game_id):
        """
        Visualize the performance of a single genome.
        
        :param network: The genome's network
        :param game_id: ID of the game that will be used for evaluation
        """
        # Create space in which game will be played
        window = pyglet.window.Window(AXIS_X * PTM,
                                      AXIS_Y * PTM,
                                      "Robot Simulator - Game {id:03d}".format(id=game_id),
                                      resizable=False,
                                      visible=True)
        window.set_location(100, 100)
        
        # Setup the requested game
        game = self.create_game(game_id)
        self.state = game.reset()[D_SENSOR_LIST]
        
        # Create the visualize-environment
        space = pymunk.Space()
        options = DrawOptions()
        
        # Draw static objects - walls
        for wall in game.walls:
            wall_shape = pymunk.Segment(space.static_body,
                                        a=wall.x * PTM,
                                        b=wall.y * PTM,
                                        radius=0.05 * PTM)  # 5cm walls
            space.add(wall_shape)
        
        # Draw static objects - target
        target_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        target_body.position = game.target * PTM
        target_shape = pymunk.Circle(body=target_body, radius=BOT_RADIUS * PTM)  # Circle with 5cm radius
        target_shape.sensor = True
        space.add(target_body, target_shape)
        
        # Init player
        m = pymunk.moment_for_circle(mass=1, inner_radius=0, outer_radius=BOT_RADIUS * PTM)
        player_body = pymunk.Body(mass=1, moment=m)
        player_body.position = game.player.pos * PTM
        player_body.angle = game.player.angle
        player_shape = pymunk.Circle(body=player_body, radius=BOT_RADIUS * PTM)
        space.add(player_body, player_shape)
        
        @window.event
        def on_draw():
            window.clear()
            space.debug_draw(options=options)
        
        def update_method(dt):
            # Query the game for the next action
            dt = dt * self.speedup
            action = self.query_net(network, [self.state])
            if self.debug:
                print("Action: lw={l}, rw={r}".format(l=round(action[0][0], 2), r=round(action[0][1], 2)))
            
            # Progress game by one step
            obs, _ = game.step_dt(dt=dt, l=action[0][0], r=action[0][1])
            self.state = obs[D_SENSOR_LIST]
            
            # Update space's player coordinates and angle
            player_body.position = game.player.pos * PTM
            player_body.angle = game.player.angle
            space.step(dt)
        
        # Run the game
        pyglet.clock.schedule_interval(update_method, 1.0 / (FPS * self.speedup))
        pyglet.app.run()
    
    def create_game(self, i):
        """
        :param i: Game-ID
        :return: Game object
        """
        return Game(game_id=i,
                    rel_path=self.rel_path,
                    silent=True)
