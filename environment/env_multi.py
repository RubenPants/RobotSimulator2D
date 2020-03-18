"""
env_multi.py

Environment where a single genome gets evaluated over multiple games. This environment will be called in a process.
"""
from config import GameConfig, NeatConfig
from environment.entities.game import get_game
from utils.dictionary import D_DONE, D_SENSOR_LIST


class MultiEnvironment:
    """ This class provides an environment to evaluate a single genome on multiple games. """
    
    __slots__ = ("batch_size", "games", "make_net", "max_steps", "query_net", "game_config", "neat_config")
    
    def __init__(self,
                 make_net,
                 query_net,
                 game_config: GameConfig,
                 neat_config: NeatConfig,
                 max_steps: int):
        """
        Create an environment in which the genomes get evaluated across different games.
        
        :param make_net: Method to create a network based on the given genome
        :param query_net: Method to evaluate the network given the current state
        :param game_config: GameConfig file for game-creation
        :param neat_config: NeatConfig file specifying how genome's network will be made
        :param max_steps: Maximum number of steps a candidate drives around in a single environment
        """
        self.batch_size = None
        self.games = None
        self.make_net = make_net
        self.max_steps = max_steps
        self.query_net = query_net
        self.game_config = game_config
        self.neat_config = neat_config
    
    def eval_genome(self,
                    genome,
                    return_dict: dict = None,
                    debug: bool = False):
        """
        Evaluate a single genome in a pre-defined game-environment.
        
        :param genome: Tuple (genome_id, genome_class)
        :param return_dict: Dictionary used to return observations corresponding the genome
        :param debug: Boolean specifying if debugging is enabled or not
        """
        genome_id, genome = genome  # Split up genome by id and genome itself
        net = self.make_net(genome=genome, config=self.neat_config, game_config=self.game_config, bs=self.batch_size)
        
        # Placeholders
        games = [get_game(g, cfg=self.game_config) for g in self.games]
        states = [g.reset()[D_SENSOR_LIST] for g in games]
        finished = [False] * self.batch_size
        
        # Start iterating the environments
        step_num = 0
        while True:
            # Check if maximum iterations is reached
            if step_num == self.max_steps: break
            
            # Determine the actions made by the agent for each of the states
            if debug:
                actions = self.query_net(net, states, debug=True, step_num=step_num)
            else:
                actions = self.query_net(net, states)
            
            # Check if each game received an action
            assert len(actions) == len(games)
            
            for i, (g, a, f) in enumerate(zip(games, actions, finished)):
                # Ignore if game has finished
                if not f:
                    # Proceed the game with one step, based on the predicted action
                    obs = g.step(l=a[0], r=a[1])
                    finished[i] = obs[D_DONE]
                    
                    # Update the candidate's current state
                    states[i] = obs[D_SENSOR_LIST]
            
            # Stop if agent reached target in all the games
            if all(finished): break
            step_num += 1
        
        # Return the final observations
        if return_dict is not None: return_dict[genome_id] = [g.close() for g in games]
    
    def trace_genome(self,
                     genome,
                     return_dict: dict = None,
                     debug: bool = False):
        """
        Get the trace of a single genome for a pre-defined game-environment.
        
        :param genome: Tuple (genome_id, genome_class)
        :param return_dict: Dictionary used to return the traces corresponding the genome-game combination
        :param debug: Boolean specifying if debugging is enabled or not
        """
        genome_id, genome = genome  # Split up genome by id and genome itself
        net = self.make_net(genome=genome, config=self.neat_config, game_config=self.game_config, bs=self.batch_size)
        
        # Placeholders
        games = [get_game(g, cfg=self.game_config) for g in self.games]
        states = [g.reset()[D_SENSOR_LIST] for g in games]
        traces = [[g.player.pos.get_tuple()] for g in games]
        
        # Start iterating the environments
        step_num = 0
        while True:
            # Check if maximum iterations is reached
            if step_num == self.max_steps: break
            
            # Determine the actions made by the agent for each of the states
            if debug:
                actions = self.query_net(net, states, debug=True, step_num=step_num)
            else:
                actions = self.query_net(net, states)
            
            # Check if each game received an action
            assert len(actions) == len(games)
            
            for i, (g, a) in enumerate(zip(games, actions)):
                # Proceed the game with one step, based on the predicted action
                obs = g.step(l=a[0], r=a[1])
                
                # Update the candidate's current state
                states[i] = obs[D_SENSOR_LIST]
                
                # Update the trace
                traces[i].append(g.player.pos.get_tuple())
            
            # Next step
            step_num += 1
        
        # Return the final observations
        if return_dict is not None: return_dict[genome_id] = traces
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    def set_games(self, games):
        """
        Set the games-set with new games.
        
        :param games: List of Game-IDs
        """
        self.games = games
        self.batch_size = len(games)
    
    def get_game_params(self):
        """Return list of all game-parameters currently in self.games."""
        return [get_game(i, cfg=self.game_config).game_params() for i in self.games]
