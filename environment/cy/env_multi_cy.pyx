"""
env_multi_cy.py

Environment where a single genome gets evaluated over multiple games. This environment will be called in a process.
"""
import numpy as np
cimport numpy as np

from environment.entities.cy.game_cy cimport get_game_cy
from utils.dictionary import D_DONE, D_SENSOR_LIST

cdef class MultiEnvironmentCy:
    """ This class provides an environment to evaluate a single genome on multiple games. """
    
    __slots__ = ("batch_size", "games", "make_net", "max_steps", "query_net", "game_config", "neat_config")
    
    def __init__(self,
                 make_net,
                 query_net,
                 game_config,
                 neat_config,
                 int max_steps):
        """
        Create an environment in which the genomes get evaluated across different games.
        
        :param make_net: Method to create a network based on the given genome
        :param query_net: Method to evaluate the network given the current state
        :param game_config: GameConfig file for game-creation
        :param neat_config: NeatConfig file specifying how genome's network will be made
        :param max_steps: Maximum number of steps a candidate drives around in a single environment
        """
        self.batch_size = 0
        self.games = []
        self.make_net = make_net
        self.max_steps = max_steps
        self.query_net = query_net
        self.game_config = game_config
        self.neat_config = neat_config
    
    cpdef void eval_genome(self,
                           genome,
                           return_dict=None,
                           ):
        """
        Evaluate a single genome in a pre-defined game-environment.
        
        :param genome: Tuple (genome_id, genome_class)
        :param return_dict: Dictionary used to return observations corresponding the genome
        """
        cdef int genome_id, step_num, max_steps
        cdef list games, states, finished
        cdef set used_sensors
        cdef np.ndarray a, actions
        cdef bint f
        
        genome_id, genome = genome  # Split up genome by id and genome itself
        net = self.make_net(genome=genome, config=self.neat_config, game_config=self.game_config, bs=self.batch_size)
        
        # Initialize the games on which the genome is tested
        games = [get_game_cy(g, cfg=self.game_config) for g in self.games]
        for g in games: g.player.set_active_sensors(set(genome.connections.keys()))  # Set active-sensors
        
        # Ask for each of the games the starting-state
        states = [g.reset()[D_SENSOR_LIST] for g in games]
        
        # Finished-state for each of the games is set to false
        finished = [False] * self.batch_size
        
        # Start iterating the environments
        step_num = 0
        while True:
            # Check if maximum iterations is reached
            if step_num == self.max_steps: break
            
            # Determine the actions made by the agent for each of the states
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
    
    cpdef void trace_genome(self,
                            genome,
                            return_dict=None,
                            ):
        """
        Get the trace of a single genome for a pre-defined game-environment.
        
        :param genome: Tuple (genome_id, genome_class)
        :param return_dict: Dictionary used to return the traces corresponding the genome-game combination
        """
        cdef int genome_id, step_num, max_steps
        cdef list games, states, traces
        cdef set used_sensors
        cdef np.ndarray a, actions
        
        genome_id, genome = genome  # Split up genome by id and genome itself
        net = self.make_net(genome=genome, config=self.neat_config, game_config=self.game_config, bs=self.batch_size)
        
        # Initialize the games on which the genome is tested
        games = [get_game_cy(g, cfg=self.game_config) for g in self.games]
        for g in games: g.player.set_active_sensors(set(genome.connections.keys()))  # Set active-sensors
        
        # Ask for each of the games the starting-state
        states = [g.reset()[D_SENSOR_LIST] for g in games]

        # Initialize the traces
        traces = [[g.player.pos.get_tuple()] for g in games]
        
        # Finished-state for each of the games is set to false
        finished = [False] * self.batch_size
        
        # Start iterating the environments
        step_num = 0
        while True:
            # Check if maximum iterations is reached
            if step_num == self.max_steps: break
            
            # Determine the actions made by the agent for each of the states
            actions = self.query_net(net, states)
            
            # Check if each game received an action
            assert len(actions) == len(games)
            
            for i, (g, a, f) in enumerate(zip(games, actions, finished)):
                # Do not advance the player if target is reached
                if f:
                    traces.append(g.player.pos.get_tuple())
                    continue
                    
                # Proceed the game with one step, based on the predicted action
                obs = g.step(l=a[0], r=a[1])
                finished[i] = obs[D_DONE]
                
                # Update the candidate's current state
                states[i] = obs[D_SENSOR_LIST]
                
                # Update the trace
                traces[i].append(g.player.pos.get_tuple())
            
            # Next step
            step_num += 1
        
        # Return the final observations
        if return_dict is not None: return_dict[genome_id] = traces
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    cpdef void set_games(self, list games):
        """
        Set the games-set with new games.
        
        :param games: List of Game-IDs
        """
        self.games = games
        self.batch_size = len(games)
    
    cpdef list get_game_params(self):
        """Return list of all game-parameters currently in self.games."""
        return [get_game_cy(i, cfg=self.game_config).game_params() for i in self.games]
