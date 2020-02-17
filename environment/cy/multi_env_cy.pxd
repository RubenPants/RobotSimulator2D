"""
multi_env.py

TODO
"""
from environment.entities.cy.game_cy cimport GameCy

cdef class MultiEnvironmentCy:
    """ This class provides an environment to evaluate a single genome on multiple games. """
    cdef public int batch_size, max_duration
    cdef public list games
    cdef public str rel_path
    cdef public make_net
    cdef public query_net
    
    cpdef void eval_genome(self, genome, config, return_dict=?, bint debug=?)
    
    cpdef GameCy create_game(self, int i)
    
    cpdef void set_games(self, list games)
