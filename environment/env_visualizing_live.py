"""
env_visualizing_live.py

The visualizer gives a live visualization of a bot's run.
"""
import pyglet
import pymunk
from pymunk.pyglet_util import DrawOptions

from environment.entities.game import Game, get_game
from environment.entities.sensors import ProximitySensor
from population.population import Population
from population.utils.network_util.feed_forward_net import FeedForwardNet
from utils.dictionary import D_DONE, D_SENSOR_LIST


class LiveVisualizer:
    """
    The visualizer will visualize the run of a single genome from the population in a game of choice. This is done by
    the use of pymunk.
    """
    
    __slots__ = (
        "speedup", "state", "finished", "time",
        "make_net", "query_net", "neat_config", "game_config",
        "debug",
    )
    
    def __init__(self,
                 pop: Population,
                 debug: bool = True,
                 speedup: float = 3):
        """
        The visualizer provides methods used to visualize the performance of a single genome.
        
        :param pop: Population object
        :param debug: Generates prints (CLI) during visualization
        :param speedup: Specifies the relative speedup the virtual environment faces towards the real world
        """
        # Visualizer specific parameters
        self.speedup = speedup
        self.state = None
        self.finished = False
        self.time = 0
        
        # Network specific parameters
        self.make_net = pop.make_net
        self.query_net = pop.query_net
        self.neat_config = pop.config
        self.game_config = pop.game_config
        
        # Debug options
        self.debug = debug
    
    # TODO: Generalize and use multiple robots?
    def visualize(self, genome, game_id: int, random_init: bool = False, random_target: bool = False):
        """
        Visualize the performance of a single genome.
        
        :param genome: Tuple (genome_id, genome_class)
        :param game_id: ID of the game that will be used for evaluation
        :param random_init: Random initial position for the agent
        :param random_target: Randomize the maze's target location
        """
        # Make the network used during visualization
        net = self.make_net(genome=genome, config=self.neat_config, game_config=self.game_config, bs=1)
        
        # Create the requested game
        game: Game = get_game(game_id, cfg=self.game_config)
        game.player.set_active_sensors(set(genome.connections.keys()))
        if random_target: game.set_target_random()
        
        # Create space in which game will be played
        window = pyglet.window.Window(game.x_axis * game.p2m,
                                      game.y_axis * game.p2m,
                                      "Robot Simulator - Game {id:03d}".format(id=game_id),
                                      resizable=False,
                                      visible=True)
        window.set_location(100, 100)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        
        # Setup the requested game
        self.state = game.reset(random_init=random_init)[D_SENSOR_LIST]
        self.finished = False
        
        # Create the visualize-environment
        space = pymunk.Space()
        options = DrawOptions()
        
        # Draw static objects - walls
        for wall in game.walls:
            wall_shape = pymunk.Segment(space.static_body,
                                        a=wall.x * game.p2m,
                                        b=wall.y * game.p2m,
                                        radius=0.05 * game.p2m)  # 5cm walls
            wall_shape.color = (0, 0, 0)
            space.add(wall_shape)
        
        # Draw static objects - target
        target_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        target_body.position = game.target * game.p2m
        target_shape = pymunk.Circle(body=target_body,
                                     radius=game.bot_radius * game.p2m)
        target_shape.sensor = True
        target_shape.color = (0, 128, 0)
        space.add(target_body, target_shape)
        
        # Init player²
        m = pymunk.moment_for_circle(mass=2,
                                     inner_radius=0,
                                     outer_radius=game.bot_radius * game.p2m)
        player_body = pymunk.Body(mass=1, moment=m)
        player_body.position = game.player.pos * game.p2m
        player_body.angle = game.player.angle
        player_shape = pymunk.Circle(body=player_body,
                                     radius=game.bot_radius * game.p2m)
        player_shape.color = (255, 0, 0)
        space.add(player_body, player_shape)
        label = pyglet.text.Label(f'{self.time}',  # TODO: Creates error in WeakMethod after run (during termination)
                                  font_size=16,
                                  color=(100, 100, 100, 100),
                                  x=window.width - 20, y=window.height - 20,
                                  anchor_x='center', anchor_y='center')
        
        # Draw the robot's sensors
        def draw_sensors():
            [space.remove(s) for s in space.shapes if s.sensor and type(s) == pymunk.Segment]
            for key in game.player.active_sensors:
                s = game.player.sensors[key]
                if type(s) == ProximitySensor:
                    line = pymunk.Segment(space.static_body,
                                          a=s.start_pos * game.p2m,
                                          b=s.end_pos * game.p2m,
                                          radius=0.5)
                    line.sensor = True
                    touch = ((s.start_pos - s.end_pos).get_length() < game.ray_distance - 0.05)
                    line.color = (100, 100, 100) if touch else (200, 200, 200)  # Brighten up ray if it makes contact
                    space.add(line)
        
        @window.event
        def on_draw():
            window.clear()
            draw_sensors()
            label.draw()
            space.debug_draw(options=options)
        
        def update_method(_):  # Input dt ignored
            dt = 1 / game.fps
            self.time += dt
            label.text = str(int(self.time))
            
            # Stop when target is reached
            if not self.finished:
                # Query the game for the next action
                action = self.query_net(net, [self.state])
                if self.debug:
                    print("Passed time:", round(dt, 3))
                    print("Location: x={}, y={}".format(
                            round(player_body.position.x / game.p2m, 2),
                            round(player_body.position.y / game.p2m, 2)))
                    print("Action: lw={l}, rw={r}".format(l=round(action[0][0], 3), r=round(action[0][1], 3)))
                    print("Observation:", [round(s, 3) for s in self.state])
                
                # Progress game by one step
                obs = game.step_dt(dt=dt, l=action[0][0], r=action[0][1])
                self.finished = obs[D_DONE]
                self.state = obs[D_SENSOR_LIST]
                
                # Update space's player coordinates and angle
                player_body.position = game.player.pos * game.p2m
                player_body.angle = game.player.angle
            space.step(dt)
        
        # Run the game
        pyglet.clock.schedule_interval(update_method, 1.0 / (game.fps * self.speedup))
        pyglet.app.run()


def used_sensor(network: FeedForwardNet, sensor_index):
    if sum([network.in2out[i][sensor_index] for i in range(len(network.in2out))]):
        return True
    elif 'in2hid' in network.__dict__ and \
            sum([network.in2hid[i][sensor_index] for i in range(len(network.in2hid))]) != 0:
        return True
    return False
