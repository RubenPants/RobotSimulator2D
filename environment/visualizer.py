"""
visualizer.py

TODO
"""

import pyglet
import pymunk
from pymunk.pyglet_util import DrawOptions

from environment.entities.game import Game
from utils.config import *
from utils.dictionary import D_SENSOR_LIST, D_DONE


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
        self.finished = False
        
        # Network specific parameters
        self.query_net = query_net
        
        # Debug options
        self.debug = debug
    
    def visualize(self, network, game_id):  # TODO: Possibility to generalize and use multiple robots?
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
        pyglet.gl.glClearColor(1, 1, 1, 1)
        
        # Setup the requested game
        game = self.create_game(game_id)
        self.state = game.reset()[D_SENSOR_LIST]
        self.finished = False
        
        # Create the visualize-environment
        space = pymunk.Space()
        options = DrawOptions()
        
        # Draw static objects - walls
        for wall in game.walls:
            wall_shape = pymunk.Segment(space.static_body,
                                        a=wall.x * PTM,
                                        b=wall.y * PTM,
                                        radius=0.05 * PTM)  # 5cm walls
            wall_shape.color = (0, 0, 0)
            space.add(wall_shape)
        
        # Draw static objects - target
        target_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        target_body.position = game.target * PTM
        target_shape = pymunk.Circle(body=target_body, radius=BOT_RADIUS * PTM)  # Circle with 5cm radius
        target_shape.sensor = True
        target_shape.color = (0, 128, 0)
        space.add(target_body, target_shape)
        
        # Init player
        m = pymunk.moment_for_circle(mass=1, inner_radius=0, outer_radius=BOT_RADIUS * PTM)
        player_body = pymunk.Body(mass=1, moment=m)
        player_body.position = game.player.pos * PTM
        player_body.angle = game.player.angle
        player_shape = pymunk.Circle(body=player_body, radius=BOT_RADIUS * PTM)
        player_shape.color = (255, 0, 0)
        space.add(player_body, player_shape)
        
        # Draw the robot's sensors
        def draw_sensors():
            [space.remove(s) for s in space.shapes if s.sensor and type(s) == pymunk.Segment]
            for s in game.player.proximity_sensors:
                line = pymunk.Segment(space.static_body,
                                      a=s.start_pos * PTM,
                                      b=s.end_pos * PTM,
                                      radius=0.5)
                line.sensor = True
                touch = ((s.start_pos - s.end_pos).get_length() < SENSOR_RAY_DISTANCE - 0.05)
                line.color = (100, 100, 100) if touch else (200, 200, 200)  # Brighten up ray if it makes contact
                space.add(line)
        
        @window.event
        def on_draw():
            window.clear()
            draw_sensors()
            space.debug_draw(options=options)
        
        def update_method(dt):
            dt = dt * self.speedup
            
            # Stop when target is reached
            if not self.finished:
                # Query the game for the next action
                action = self.query_net(network, [self.state])
                if self.debug:
                    print("Passed time:", round(dt, 3))
                    print("Location: x={}, y={}".format(round(player_body.position.x / PTM, 2),
                                                        round(player_body.position.y / PTM, 2)))
                    print("Action: lw={l}, rw={r}".format(l=round(action[0][0], 3), r=round(action[0][1], 3)))
                    print("Observation:", [round(s, 3) for s in self.state])
                
                # Progress game by one step
                obs = game.step_dt(dt=dt, l=action[0][0], r=action[0][1])
                self.finished = obs[D_DONE]
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
