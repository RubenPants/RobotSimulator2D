"""
genome_config.py

Handles the configuration of the genomes.
"""
from __future__ import division, print_function

from itertools import count

from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.config import ConfigParameter, write_pretty_params
from neat.six_util import iterkeys


class DefaultGenomeConfig(object):
    """Sets up and holds configuration information for the DefaultGenome class."""
    
    allowed_connectivity = ['unconnected', 'fs_neat_nohidden', 'fs_neat', 'fs_neat_hidden',
                            'full_nodirect', 'full', 'full_direct',
                            'partial_nodirect', 'partial', 'partial_direct']
    
    def __init__(self, params):
        self.num_outputs = None  # Placeholder
        self.num_inputs = None  # Placeholder
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs
        
        self._params = [
            ConfigParameter('num_inputs', int),
            ConfigParameter('num_outputs', int),
            ConfigParameter('num_hidden', int),
            ConfigParameter('compatibility_disjoint_coefficient', float),
            ConfigParameter('compatibility_weight_coefficient', float),
            ConfigParameter('conn_add_prob', float),
            ConfigParameter('conn_delete_prob', float),
            ConfigParameter('node_add_prob', float),
            ConfigParameter('node_delete_prob', float),
            ConfigParameter('structural_mutation_surer', str, 'default'),
            ConfigParameter('initial_connection', str, 'unconnected'),
            ConfigParameter('enable_gru', bool)
        ]
        
        # Gather configuration data from the gene classes.
        self.node_gene_type = params['node_gene_type']
        self._params += self.node_gene_type.get_config_params()
        self.output_node_gene_type = params['output_node_gene_type']
        # self._params += self.output_node_gene_type.get_config_params()  # Identical to self.node_gene_type.get_config
        self.gru_node_gene_type = params['gru_node_gene_type']
        # gru_node_gene_type must not be added to _params (has no relevant 'get_config_params()')
        self.connection_gene_type = params['connection_gene_type']
        self._params += self.connection_gene_type.get_config_params()
        
        # Use the configuration data to interpret the supplied parameters.
        for p in self._params: setattr(self, p.name, p.interpret(params))
        
        # By convention, input pins have negative keys, and the output pins have the first keys (0,1,...)
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]
        
        self.connection_fraction = None
        
        # Verify that initial connection type is valid.
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError("'partial' connection value must be between 0.0 and 1.0, inclusive.")
        
        assert self.initial_connection in self.allowed_connectivity
        
        # Verify structural_mutation_surer is valid.
        if self.structural_mutation_surer.lower() in ['1', 'yes', 'true', 'on']:
            self.structural_mutation_surer = 'true'
        elif self.structural_mutation_surer.lower() in ['0', 'no', 'false', 'off']:
            self.structural_mutation_surer = 'false'
        elif self.structural_mutation_surer.lower() == 'default':
            self.structural_mutation_surer = 'default'
        else:
            error_string = f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}"
            raise RuntimeError(error_string)
        
        self.node_indexer = None
    
    def __str__(self):
        """Readable format of the genome configuration."""
        attrib = [a.name for a in self._params]
        result = "Default genome configuration:"
        for a in attrib:
            attr = getattr(self, a)
            if isinstance(attr, float):
                result += f"\n\t- {a} = {round(attr, 3)}"
            else:
                result += f"\n\t- {a} = {attr}"
        return result
    
    def add_activation(self, name, func):
        self.activation_defs.add(name, func)
    
    def add_aggregation(self, name, func):
        self.aggregation_function_defs.add(name, func)
    
    def save(self, f):
        if 'partial' in self.initial_connection:
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError("'partial' connection value must be between 0.0 and 1.0, inclusive.")
            f.write(f'initial_connection      = {self.initial_connection} {self.connection_fraction}\n')
        else:
            f.write(f'initial_connection      = {self.initial_connection}\n')
        
        assert self.initial_connection in self.allowed_connectivity
        
        write_pretty_params(f, self, [p for p in self._params if 'initial_connection' not in p.name])
    
    def get_new_node_key(self, node_dict):
        if self.node_indexer is None: self.node_indexer = count(max(list(iterkeys(node_dict))) + 1)
        new_id = next(self.node_indexer)
        assert new_id not in node_dict
        return new_id
    
    def check_structural_mutation_surer(self):
        """
        If structural_mutation_surer evaluates to True, then an attempt to add a node to a genome lacking connections
        will result in adding a connection instead; furthermore, if an attempt to add a connection tries to add a
        connection that already exists, that connection will be enabled.
        This defaults to "default".
        """
        if self.structural_mutation_surer == 'true':
            return True
        elif self.structural_mutation_surer == 'false' or self.structural_mutation_surer == 'default':
            return False
        else:
            raise RuntimeError(f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}")
