"""
genome_config.py

Configuration file corresponding the creation and mutation of the genomes.
"""
from itertools import count

from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.six_util import iterkeys

from configs.base_config import BaseConfig
from utils.dictionary import *


class GenomeConfig(BaseConfig):
    """Genome-specific configuration parameters."""
    
    __slots__ = {
        'activation_default', 'activation_defs', 'activation_mutate_rate', 'activation_options', 'aggregation_default',
        'aggregation_defs', 'aggregation_mutate_rate', 'aggregation_options', 'bias_init_mean', 'bias_init_stdev',
        'bias_max_value', 'bias_min_value', 'bias_mutate_power', 'bias_mutate_rate', 'bias_replace_rate',
        'compatibility_disjoint_coefficient', 'compatibility_weight_coefficient', 'conn_add_prob', 'conn_delete_prob',
        'conn_fraction', 'enabled_default', 'enabled_mutate_rate', 'gru_enabled', 'gru_init_mean',
        'gru_init_stdev', 'gru_max_value', 'gru_min_value', 'gru_mutate_power', 'gru_mutate_rate', 'gru_node_prob',
        'gru_replace_rate', 'initial_connection', 'keys_input', 'node_add_prob', 'node_delete_prob', 'node_indexer',
        'num_inputs', 'num_outputs', 'keys_output', 'weight_init_mean', 'weight_init_stdev', 'weight_max_value',
        'weight_min_value', 'weight_mutate_power', 'weight_mutate_rate', 'weight_replace_rate'
    }
    
    allowed_connectivity = ['unconnected', 'fs_neat_nohidden', 'fs_neat', 'fs_neat_hidden',
                            'full_nodirect', 'full', 'full_direct',
                            'partial_nodirect', 'partial', 'partial_direct']
    
    def __init__(self):
        # Initial node activation function  [def=D_GELU]
        self.activation_default: str = D_GELU
        # List of all supported activation functions   [def=/]
        self.activation_defs = ActivationFunctionSet()
        # Probability of changing the activation function  [def=0]
        self.activation_mutate_rate: float = 0.0
        # All possible activation functions between whom can be switched during mutation  [def=D_GELU]
        self.activation_options: str = D_GELU
        # The default aggregation function attribute assigned to new nodes  [def=D_SUM]
        self.aggregation_default: str = D_SUM
        # List of all supported aggregation functions  [def=/]
        self.aggregation_defs = AggregationFunctionSet()
        # Probability of mutating towards another aggregation_option  [def=0]
        self.aggregation_mutate_rate: float = 0.0
        # Aggregation options between whom can be mutated  [def=D_SUM]
        self.aggregation_options: str = D_SUM
        # The mean of the gaussian distribution, used to select the bias attribute values for new nodes  [def=0]
        self.bias_init_mean: float = 0.0
        # Standard deviation of gaussian distribution, used to select the bias attribute values of new nodes  [def=1]
        self.bias_init_stdev: float = 1.0
        # The maximum allowed bias value, biases above this threshold will be clamped to this value  [def=2]
        self.bias_max_value: float = 2.0
        # The minimum allowed bias value, biases below this threshold will be clamped to this value  [def=-2]
        self.bias_min_value: float = -2.0
        # The standard deviation of the zero-centered gaussian from which a bias value mutation is drawn  [def=0.2] TODO
        self.bias_mutate_power: float = 0.2
        # The probability that mutation will change the bias of a node by adding a random value  [def=0.2]  TODO
        self.bias_mutate_rate: float = 0.2
        # The probability that mutation will replace the bias of a node with a completely random value  [def=0.05]
        self.bias_replace_rate: float = 0.05
        # Full weight of disjoint and excess nodes on determining genomic distance  [def=1.0]  # TODO: Separate for GRU?
        self.compatibility_disjoint_coefficient: float = 1.0
        # Coefficient for each weight or bias difference contribution to the genomic distance  [def=0.5]
        self.compatibility_weight_coefficient: float = 0.5
        # Probability of adding a connection between existing nodes during mutation (each generation)  [def=0.1]  TODO
        self.conn_add_prob: float = 0.1
        # Probability of deleting an existing connection during mutation (each generation)  [def=0.1]  TODO
        self.conn_delete_prob: float = 0.1
        # Denotes to which fraction the initial connectivity holds, parsed from self.initial_connection  [def=/]
        self.conn_fraction: float = 0
        # Enable the algorithm to disable (and re-enable) existing connections  [def=True]
        self.enabled_default: bool = True
        # The probability that mutation will replace the 'enabled status' of a connection  [def=0.05]
        self.enabled_mutate_rate: float = 0.05
        # Enable the genomes to mutate GRU nodes  [def=True]  TODO
        self.gru_enabled: bool = True
        # Mean of the gaussian distribution used to select the GRU attribute values  [def=0]
        self.gru_init_mean: float = 0.0
        # Standard deviation of the gaussian used to select the GRU attributes values  [def=1]
        self.gru_init_stdev: float = 1.0
        # The maximum allowed GRU value, values above this will be clipped  [def=2]
        self.gru_max_value: float = 2.0
        # The minimum allowed GRU value, values below this will be clipped  [def=-2]
        self.gru_min_value: float = -2.0
        # The standard deviation of the zero-centered gaussian from which a GRU value mutation is drawn  [def=0.05]
        self.gru_mutate_power: float = 0.05
        # Probability of a GRU value to mutate  [def=0.2]  TODO
        self.gru_mutate_rate: float = 0.2
        # Probability of mutating a GRU node rather than a simple node  [def=0.6]  TODO
        self.gru_node_prob: float = 1
        # Probability of assigning (single) random value in GRU, based on gru_init_mean and gru_init_stdev  [def=0.05]
        self.gru_replace_rate: float = 0.05
        # Initial connectivity of newly-created genomes  [def=D_PARTIAL_DIRECT_05]  TODO
        self.initial_connection = D_FULL_NODIRECT
        # Input-keys, which are by convention negative starting from -1 and descending, set in update()  [def=/]
        self.keys_input = None
        # Output-keys, which start by convention from 0 and increment with each output, set in update()  [def=/]
        self.keys_output = None
        # Probability of adding a node during mutation (each generation)  [def=0.05]  TODO
        self.node_add_prob: float = 0.05
        # Probability of removing a node during mutation (each generation)  [def=0.05]  TODO
        self.node_delete_prob: float = 0.05
        # Node-indexer helps with the generation of node-keys  [def=/]
        self.node_indexer = None
        # Number of inputs, which are the robot's sensors  [def=/]
        self.num_inputs: int = 0
        # Number of output nodes, which are the wheels: [left_wheel, right_wheel]  [def=2]
        self.num_outputs: int = 2
        # Mean of the gaussian distribution used to select the weight attribute values for new connections  [def=0]
        self.weight_init_mean: float = 0.0
        # Standard deviation of the gaussian used to select the weight attributes values for new connections  [def=1]
        self.weight_init_stdev: float = 1.0
        # The maximum allowed weight value, weights above this value will be clipped to this value  [def=2]
        self.weight_max_value: float = 2.0
        # The minimum allowed weight value, weights below this value will be clipped to this value  [def=-2]
        self.weight_min_value: float = -2.0
        # The standard deviation of the zero-centered gaussian from which a weight value mutation is drawn [def=0.2]TODO
        self.weight_mutate_power: float = 0.2
        # Probability of a weight (connection) to mutate  [def=0.2]  TODO
        self.weight_mutate_rate: float = 0.2
        # Probability of assigning completely new value, based on weight_init_mean and weight_init_stdev  [def=0.05]
        self.weight_replace_rate: float = 0.05
    
    def update(self, main_config):
        """Reload the current number of input sensors."""
        from environment.entities.robots import get_number_of_sensors
        self.num_inputs: int = get_number_of_sensors(cfg=main_config.bot)
        self.keys_input = [-i - 1 for i in range(self.num_inputs)]
        self.keys_output = [i for i in range(self.num_outputs)]
        
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.conn_fraction = float(p)
            if not (0 <= self.conn_fraction <= 1):
                raise RuntimeError("'partial' connection value must be between 0.0 and 1.0, inclusive.")
        assert self.initial_connection in self.allowed_connectivity
    
    def add_activation(self, name, func):
        self.activation_defs.add(name, func)
    
    def add_aggregation(self, name, func):
        self.aggregation_defs.add(name, func)
    
    def get_new_node_key(self, node_dict):
        if self.node_indexer is None: self.node_indexer = count(max(list(iterkeys(node_dict))) + 1)
        new_id = next(self.node_indexer)
        assert new_id not in node_dict
        return new_id
