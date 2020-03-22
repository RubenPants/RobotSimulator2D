"""
env_multi_cy.py

Environment where a single genome gets evaluated over multiple games. This environment will be called in a process.
"""


cdef class MultiEnvironmentCy:
    """ This class provides an environment to evaluate a single genome on multiple games. """
    cdef public int batch_size, max_steps
    cdef public list games
    cdef public str rel_path
    cdef public make_net
    cdef public query_net
    cdef public game_config
    cdef public neat_config
    
    cpdef void eval_genome(self, genome, return_dict=?, bint random_init=?)
    
    cpdef void trace_genome(self, genome, return_dict=?, bint random_init=?)

    cpdef void set_games(self, list games)
    
    cpdef list get_game_params(self)
