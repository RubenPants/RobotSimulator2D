"""
multi_env.py

TODO
"""
import sys

from utils.dictionary import D_SENSOR_LIST, D_DONE

if sys.platform == 'linux':
    from environment.entities.cy.game_cy import GameCy
else:
    from environment.entities.game import Game


class MultiEnvironment:
    """ This class provides an environment to evaluate a single genome on multiple games. """
    
    def __init__(self,
                 make_net,
                 query_net,
                 max_duration: int = 100,
                 rel_path: str = ''):
        """
        Create an environment in which the genomes get evaluated across different games.
        
        :param make_net: Method to create a network based on the given genome
        :param query_net: Method to evaluate the network given the current state
        :param max_duration: Maximum number of seconds a candidate drives around in a single environment
        :param rel_path: Relative path pointing to the 'environment/' folder
        """
        self.batch_size = None
        self.games = None
        self.make_net = make_net
        self.max_steps = max_duration * FPS
        self.query_net = query_net
        self.rel_path = rel_path
    
    def eval_genome(self,
                    genome,
                    config,
                    return_dict: dict = None,
                    debug: bool = False):
        """
        Evaluate a single genome in a pre-defined game-environment.
        
        :param genome: Tuple (genome_id, genome_class)
        :param config: Config file specifying how genome's network will be made
        :param return_dict: Dictionary used to return observations corresponding the genome
        :param debug: Boolean specifying if debugging is enabled or not
        """
        genome_id, genome = genome  # Split up genome by id and genome itself
        net = self.make_net(genome, config, self.batch_size)
        
        # Placeholders
        games = [self.create_game(g) for g in self.games]
        states = [g.reset()[D_SENSOR_LIST] for g in games]
        finished = [False] * self.batch_size
        
        # Start iterating the environments
        step_num = 0
        while True:
            # Check if maximum iterations is reached
            if self.max_steps is not None and step_num == self.max_steps:
                break
            
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
            if all(finished):
                break
            step_num += 1
        
        # Return the final observations
        if return_dict is not None:
            return_dict[genome_id] = [g.close() for g in games]
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    def create_game(self, i):
        """
        :param i: Game-ID
        :return: Game or GameCy object
        """
        if sys.platform == 'linux':
            return GameCy(game_id=i,
                          rel_path=self.rel_path,
                          silent=True)
        else:
            return Game(game_id=i,
                        rel_path=self.rel_path,
                        silent=True)
    
    def set_games(self, games):
        """
        Set the games-set with new games.
        
        :param games: List of Game-IDs
        """
        self.games = games
        self.batch_size = len(games)
