"""
multi_env.py

TODO
"""
import numpy as np
cimport numpy as np
from environment.entities.cy.game_cy cimport GameCy
from utils.dictionary import D_DONE, D_SENSOR_LIST

cdef class MultiEnvironmentCy:
    """ This class provides an environment to evaluate a single genome on multiple games. """
    
    __slots__ = ("batch_size", "games", "make_net", "max_steps", "query_net")
    
    def __init__(self,
                 make_net,
                 query_net,
                 int max_steps):
        """
        Create an environment in which the genomes get evaluated across different games.
        
        :param make_net: Method to create a network based on the given genome
        :param query_net: Method to evaluate the network given the current state
        :param max_steps: Maximum number of steps a candidate drives around in a single environment
        """
        self.batch_size = 0
        self.games = []
        self.make_net = make_net
        self.max_steps = max_steps
        self.query_net = query_net
    
    cpdef void eval_genome(self,
                           genome,
                           config,
                           return_dict=None,
                           bint debug=False):
        """
        Evaluate a single genome in a pre-defined game-environment.
        
        :param genome: Tuple (genome_id, genome_class)
        :param config: Config file specifying how genome's network will be made
        :param return_dict: Dictionary used to return observations corresponding the genome
        :param debug: Boolean specifying if debugging is enabled or not
        """
        cdef int genome_id, step_num, max_steps
        cdef list games, states, finished
        cdef np.ndarray a, actions
        cdef bint f
        
        # Split up genome by id and genome itself
        genome_id, genome = genome
        net = self.make_net(genome, config, self.batch_size)
        
        # Placeholders
        games = [self.create_game(g) for g in self.games]
        states = [g.reset()[D_SENSOR_LIST] for g in games]
        finished = [False] * self.batch_size
        
        # Start iterating the environments
        step_num = 0
        while True:
            # Check if maximum iterations is reached
            if step_num == self.max_steps: break
            
            # Determine the actions made by the agent for each of the states
            if debug: actions = self.query_net(net, states, debug=True, step_num=step_num)
            else: actions = self.query_net(net, states)
            
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
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    cpdef GameCy create_game(self, int i):
        """
        :param i: Game-ID
        :return: Game or GameCy object
        """
        return GameCy(game_id=i,
                    silent=True)
    
    cpdef void set_games(self, list games):
        """
        Set the games-set with new games.
        
        :param games: List of Game-IDs
        """
        self.games = games
        self.batch_size = len(games)
        
    cpdef list get_game_params(self):
        """Return list of all game-parameters currently in self.games."""
        return [self.create_game(i).game_params() for i in self.games]
