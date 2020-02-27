"""
genes.py

Handles node and connection genes.
"""
import warnings
from random import random

import numpy as np
import torch

from population.utils.genome_util.attributes import BiasAttribute, BoolAttribute, FloatAttribute, StringAttribute, \
    WeightAttribute
from utils.dictionary import D_ACTIVATION, D_TANH


class BaseGene(object):
    """
    Handles functions shared by multiple types of genes (both node and connection), including crossover and calling
    mutation methods.
    """
    
    _gene_attributes = None
    
    def __init__(self, key):
        """Key to identify the gene."""
        self.key = key
    
    def __str__(self):
        attrib = ['key'] + [a.name for a in self._gene_attributes]
        body = []
        for a in attrib:
            attr = getattr(self, a)
            if isinstance(attr, float):
                body.append(f"{a}={round(attr, 3)}")
            else:
                body.append(f"{a}={attr}")
        return f'{self.__class__.__name__}({", ".join(body)})'
    
    def __lt__(self, other):
        """Used to sort the genes."""
        assert isinstance(self.key, type(other.key)), f"Cannot compare keys {self.key!r} and {other.key!r}"
        return self.key < other.key
    
    @classmethod
    def parse_config(cls, config, param_dict):
        pass
    
    @classmethod
    def get_config_params(cls):
        """Get a list of all the configuration parameters (stored in the gene's attributes)."""
        params = []
        if not hasattr(cls, '_gene_attributes'):
            setattr(cls, '_gene_attributes', getattr(cls, '__gene_attributes__'))
            warnings.warn(f"Class '{cls.__name__:!s}' {cls:!r} needs '_gene_attributes' not '__gene_attributes__'",
                          DeprecationWarning)
        for a in cls._gene_attributes: params += a.get_config_params()
        return params
    
    def init_attributes(self, config):
        """ Set the initial attributes as claimed by the config. """
        for a in self._gene_attributes: setattr(self, a.name, a.init_value(config))
    
    def mutate(self, config):
        """ Perform the mutation operation. """
        for a in self._gene_attributes:
            v = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))
    
    def copy(self):
        """ Copy the gene (this class). """
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes: setattr(new_gene, a.name, getattr(self, a.name))
        return new_gene
    
    def crossover(self, gene2):
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == gene2.key
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            if random() > 0.5:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(gene2, a.name))
        
        return new_gene


class DefaultNodeGene(BaseGene):
    """Default node configuration, as specified by the Python-NEAT documentation."""
    
    _gene_attributes = [FloatAttribute('bias'),
                        StringAttribute('activation', options='relu'),
                        StringAttribute('aggregation', options='sum')]
    
    def __init__(self, key):
        # Placeholders
        self.bias = None
        self.activation = None
        self.aggregation = None
        
        assert isinstance(key, int), f"DefaultNodeGene key must be an int, not {key!r}"
        BaseGene.__init__(self, key)
    
    def distance(self, other, config):
        d = abs(self.bias - other.bias)
        if self.key not in [0, 1] and other.key not in [0, 1]:  # Exclude comparison with activation of output nodes
            if self.activation != other.activation: d += 1.0
        if self.aggregation != other.aggregation: d += 1.0  # Normally, aggregation is always equal to 'sum'
        return d * config.compatibility_weight_coefficient


class OutputNodeGene(DefaultNodeGene):
    """Node representation for each of the network's outputs."""
    
    def __init__(self, key):
        assert isinstance(key, int), f"OutputNodeGene key must be an int, not {key!r}"
        super().__init__(key)
    
    def init_attributes(self, config):
        """ Set the initial attributes as claimed by the config, but force activation to be tanh """
        for a in self._gene_attributes:
            if a.name == D_ACTIVATION:
                setattr(self, a.name, D_TANH)  # Hard-coded output
            else:
                setattr(self, a.name, a.init_value(config))
    
    def mutate(self, config):
        """ Perform the mutation operation. Prevent mutation to happen on the output's activation. """
        for a in self._gene_attributes:
            v = getattr(self, a.name)
            if a.name != D_ACTIVATION: setattr(self, a.name, a.mutate_value(v, config))


class GruNodeGene(BaseGene):  # TODO: Mutate functionality
    """Custom GRU cell implementation."""
    
    _gene_attributes = [BiasAttribute('bias_ih'),
                        BiasAttribute('bias_hh'),
                        WeightAttribute('weight_ih'),
                        WeightAttribute('weight_hh')]
    
    def __init__(self, key, input_keys=None):
        # Placeholders
        self.input_keys = input_keys if input_keys else []
        self.hidden_size = 1  # TODO: Generalize such that output can be extended (i.e. vector with specific values for each output)
        self.weight_ih = None
        self.weight_hh = None
        self.bias_ih = None
        self.bias_hh = None
        self.bias = 0  # No external bias applied
        assert isinstance(key, int), f"OutputNodeGene key must be an int, not {key!r}"
        BaseGene.__init__(self, key)
    
    def __str__(self):
        attrib = ['key'] + [a.name for a in self._gene_attributes]
        body = []
        for a in attrib:
            attr = getattr(self, a)
            if isinstance(attr, torch.Tensor):
                body.append(f"{a}={np.asarray(attr.tolist()).round(3).tolist()}")
            else:
                body.append(f"{a}={attr}")
        return f'{self.__class__.__name__}({", ".join(body)})'
    
    def init_attributes(self, config):
        setattr(self, 'bias_ih', self._gene_attributes[0].init_value(config, self.hidden_size))
        setattr(self, 'bias_hh', self._gene_attributes[1].init_value(config, self.hidden_size))
        # By default will the initial input_size=1 (i.e. node added on top of one connection)
        setattr(self, 'weight_ih', self._gene_attributes[2].init_value(config, self.hidden_size, len(
                self.input_keys)))  # TODO: Handle init connections!
        setattr(self, 'weight_hh', self._gene_attributes[2].init_value(config, self.hidden_size, self.hidden_size))
    
    def get_gru(self):
        """Return a PyTorch GRUCell based on current configuration."""
        gru = torch.nn.GRUCell(input_size=len(self.input_keys), hidden_size=self.hidden_size)
        gru.weight_ih[:] = self.weight_ih
        gru.weight_hh[:] = self.weight_hh
        gru.bias_ih[:] = self.bias_ih
        gru.bias_hh[:] = self.bias_hh
        return gru
    
    def append_key(self, config, k):
        """Extend the input-key list with the given key, and expand the corresponding weights."""
        if k not in self.input_keys:
            # Find the index to insert the key
            lst = [i for i in range(len(self.input_keys)) if self.input_keys[i] > 3]
            idx = lst[0] if lst else len(self.input_keys)
            
            # Save key to list
            self.input_keys.insert(idx, k)
            
            # Update weight_ih correspondingly by inserting random initialized tensor in correct position
            new_tensor = WeightAttribute('temp').init_value(config, hidden_size=self.hidden_size, input_size=1)
            self.weight_ih[:] = torch.cat((self.weight_ih[:, :idx], new_tensor, self.weight_ih[:, idx:]), dim=1)
        
    def delete_key(self, k):
        """Delete one of the input-keys with its corresponding weights."""
        # Check if key in keys
        if k in self.input_keys:
            idx = self.input_keys.index(k)
            self.input_keys.remove(k)
            
            # Remove corresponding weight-column
            self.weight_ih[:] = torch.cat((self.weight_ih[:, :idx], self.weight_ih[:, idx+1:]),dim=1)


class DefaultConnectionGene(BaseGene):
    """Default connection configuration, as specified by the Python-NEAT documentation."""
    
    _gene_attributes = [FloatAttribute('weight'),
                        BoolAttribute('enabled')]
    
    def __init__(self, key):
        # Placeholders
        self.enabled = None
        self.weight = None
        
        assert isinstance(key, tuple), f"DefaultConnectionGene key must be a tuple, not {key!r}"
        BaseGene.__init__(self, key)
    
    def distance(self, other, config):
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled: d += 1.0
        return d * config.compatibility_weight_coefficient
