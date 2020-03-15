"""
genes.py

Handles node and connection genes.
"""
import copy
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
    
    _attributes = None
    
    def __init__(self, key):
        """Key to identify the gene."""
        self.key = key
    
    def __str__(self):
        attrib = ['key'] + [a.name for a in self._attributes]
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
        if not hasattr(cls, '_attributes'):
            setattr(cls, '_attributes', getattr(cls, '__attributes__'))
            warnings.warn(f"Class '{cls.__name__:!s}' {cls:!r} needs '_attributes' not '__attributes__'",
                          DeprecationWarning)
        for a in cls._attributes: params += a.get_config_params()
        return params
    
    def init_attributes(self, config):
        """ Set the initial attributes as claimed by the config. """
        for a in self._attributes: setattr(self, a.name, a.init_value(config))
    
    def mutate(self, config):
        """ Perform the mutation operation. """
        for a in self._attributes:
            v = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))
    
    def copy(self):
        """ Copy the gene (this class). """
        new_gene = self.__class__(self.key)
        for a in self._attributes: setattr(new_gene, a.name, getattr(self, a.name))
        return new_gene
    
    def crossover(self, other, ratio):
        """
        Creates a new gene randomly inheriting attributes from its parents.
        
        :param other: Other parent-gene
        :param ratio: Likelihood of using the current gene (float in [0,1])
        """
        assert self.key == other.key
        new_gene = self.__class__(self.key)
        for a in self._attributes:
            if random() <= ratio:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(other, a.name))
        
        return new_gene


class DefaultNodeGene(BaseGene):
    """Default node configuration, as specified by the Python-NEAT documentation."""
    
    _attributes = [FloatAttribute('bias'),
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
        for a in self._attributes:
            if a.name == D_ACTIVATION:
                setattr(self, a.name, D_TANH)  # Hard-coded output
            else:
                setattr(self, a.name, a.init_value(config))
    
    def mutate(self, config):
        """ Perform the mutation operation. Prevent mutation to happen on the output's activation. """
        for a in self._attributes:
            v = getattr(self, a.name)
            if a.name != D_ACTIVATION: setattr(self, a.name, a.mutate_value(v, config))


class GruNodeGene(BaseGene):
    """Custom GRU cell implementation."""
    
    _attributes = [BiasAttribute('bias_ih'),
                   BiasAttribute('bias_hh'),
                   WeightAttribute('full_weight_ih'),
                   WeightAttribute('weight_hh')]
    
    def __init__(self, key, input_keys=None):
        # Placeholders
        self.input_keys = input_keys if input_keys else []
        self.full_input_keys = input_keys if input_keys else []  # Full set of all seen input-keys
        self.hidden_size = 1  # TODO: Generalize to output vector
        self.full_weight_ih = None  # Full weight-vector mapping from full_input_set
        self.weight_ih = None  # Part of full_weight_ih that relates to current input_keys
        self.weight_hh = None
        self.bias_ih = None
        self.bias_hh = None
        self.bias = 0  # No external bias applied
        assert isinstance(key, int), f"OutputNodeGene key must be an int, not {key!r}"
        BaseGene.__init__(self, key)
    
    def __str__(self):
        attrib = ['key', 'input_keys'] + [a.name for a in self._attributes]
        body = []
        for a in attrib:
            attr = getattr(self, a)
            if isinstance(attr, torch.Tensor):
                body.append(f"{a}={np.asarray(attr.tolist()).round(3).tolist()}")
            else:
                body.append(f"{a}={attr}")
        return f'{self.__class__.__name__}({", ".join(body)})'
    
    def init_attributes(self, config):
        setattr(self, 'bias_ih', self._attributes[0].init_value(config, self.hidden_size))
        setattr(self, 'bias_hh', self._attributes[1].init_value(config, self.hidden_size))
        setattr(self, 'full_weight_ih', self._attributes[2].init_value(config, self.hidden_size, len(self.input_keys)))
        setattr(self, 'weight_hh', self._attributes[3].init_value(config, self.hidden_size, self.hidden_size))
    
    def copy(self):
        """ Copy the gene (this class). """
        new_gene: GruNodeGene = super().copy()
        
        # Other fields not stored in _attributes
        new_gene.input_keys = copy.deepcopy(self.input_keys)
        new_gene.full_input_keys = copy.deepcopy(self.full_input_keys)
        # new_gene.hidden_size = copy.deepcopy(self.hidden_size)  # TODO: only needed if generalized
        
        return new_gene
    
    def mutate(self, config):
        """ Perform the mutation operation. """
        for a in self._attributes:
            v = getattr(self, a.name)
            if a.name == 'full_weight_ih':
                mapping = [k in self.input_keys for k in self.full_input_keys]
                setattr(self, a.name, a.mutate_value(v, config, mapping))
            else:
                setattr(self, a.name, a.mutate_value(v, config))
    
    def crossover(self, other, ratio):
        """
        Creates a new gene inheriting attributes from its parents.
        
        TODO: The crossover at this (GRU) node is quite biased; only the inputs weights are to be crossed between the
         two parents, with the remaining weights only inheriting from the first parent.
        
        :param other: Other parent-gene
        :param ratio: Likelihood of using the current gene (float in [0,1])
        """
        assert self.key == other.key  # Crossover only happens at own nodes
        new_gene = self.__class__(self.key)  # Initialize empty node (used as container for crossover result)
        new_gene.full_input_keys = sorted(set(self.full_input_keys + other.full_input_keys))
        new_full_weight_ih = torch.tensor(
                np.zeros((3 * self.hidden_size, len(new_gene.full_input_keys))),
                dtype=torch.float64,
        )
        
        # Add weights column by column to the new gene
        for i, k in enumerate(new_gene.full_input_keys):
            if k in self.full_input_keys and k in other.full_input_keys:  # Shared by both
                if random() <= ratio:
                    new_full_weight_ih[:, i] = self.full_weight_ih[:, self.full_input_keys.index(k)]
                else:
                    new_full_weight_ih[:, i] = other.full_weight_ih[:, other.full_input_keys.index(k)]
            elif k in self.full_input_keys:  # Only in first parent (self)
                new_full_weight_ih[:, i] = self.full_weight_ih[:, self.full_input_keys.index(k)]
            else:  # Only in second parent (other)
                new_full_weight_ih[:, i] = other.full_weight_ih[:, other.full_input_keys.index(k)]
        
        # Assign all the parameters to the new_gene
        for a in self._attributes:
            setattr(new_gene, a.name, new_full_weight_ih if a.name == 'full_weight_ih' else getattr(self, a.name))
        
        return new_gene
    
    def distance(self, other, config):
        """Calculate the distance between two GRU nodes, which is determined by its coefficients."""
        d = 0
        d += np.linalg.norm(self.bias_ih - other.bias_ih)
        d += np.linalg.norm(self.bias_hh - other.bias_hh)
        d += np.linalg.norm(self.weight_hh - other.weight_hh)
        
        # Compare only same input keys
        key_set = sorted(set(self.input_keys + other.input_keys))
        s = np.zeros((3 * self.hidden_size, len(key_set)))
        o = np.zeros((3 * self.hidden_size, len(key_set)))
        for i, k in enumerate(key_set):
            if k in self.input_keys: s[:, i] = self.full_weight_ih[:, self.full_input_keys.index(k)]
            if k in other.input_keys: o[:, i] = other.full_weight_ih[:, other.full_input_keys.index(k)]
        d += np.linalg.norm(s - o)
        return d * config.compatibility_weight_coefficient
    
    def update_weight_ih(self):
        """Update weight_ih to be conform with the current input_keys-set."""
        self.weight_ih = torch.tensor(np.zeros((3 * self.hidden_size, len(self.input_keys))), dtype=torch.float64)
        for i, k in enumerate(self.input_keys):
            self.weight_ih[:, i] = self.full_weight_ih[:, self.full_input_keys.index(k)]
    
    def get_gru(self, weight_map=None):
        """Return a PyTorch GRUCell based on current configuration."""
        self.update_weight_ih()
        if weight_map is not None:
            gru = torch.nn.GRUCell(input_size=len(weight_map[weight_map]), hidden_size=self.hidden_size)
            gru.weight_ih[:] = self.weight_ih[:, weight_map]
        else:
            gru = torch.nn.GRUCell(input_size=len(self.input_keys), hidden_size=self.hidden_size)
            gru.weight_ih[:] = self.weight_ih
        gru.weight_hh[:] = self.weight_hh
        gru.bias_ih[:] = self.bias_ih
        gru.bias_hh[:] = self.bias_hh
        return gru
    
    def add_input(self, config, k):
        """Extend the input-key list with the given key, and expand the corresponding weights."""
        # Update self.full_weight_ih if key never seen before
        if k not in self.full_input_keys:
            # Find the index to insert the key
            lst = [i + 1 for i in range(len(self.full_input_keys)) if self.full_input_keys[i] < k]  # List of indices
            i = lst[-1] if lst else 0  # Index to insert key in
            
            # Save key to list
            self.full_input_keys.insert(i, k)
            
            # Update full_weight_ih correspondingly by inserting random initialized tensor in correct position
            new_tensor = WeightAttribute('temp').init_value(config, hidden_size=self.hidden_size, input_size=1)
            self.full_weight_ih = torch.cat((self.full_weight_ih[:, :i], new_tensor, self.full_weight_ih[:, i:]), dim=1)
        
        # Update input_keys (current key-set) analogously
        if k not in self.input_keys:
            lst = [i + 1 for i in range(len(self.input_keys)) if self.input_keys[i] < k]  # List of indices
            i = lst[-1] if lst else 0
            self.input_keys.insert(i, k)
    
    def remove_input(self, k):
        """Delete one of the input_keys, full_input_keys and full_weight_ih are left unchanged."""
        if k in self.input_keys: self.input_keys.remove(k)


class DefaultConnectionGene(BaseGene):
    """Default connection configuration, as specified by the Python-NEAT documentation."""
    
    _attributes = [FloatAttribute('weight'),
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
    
    def mutate(self, config):
        """
        Perform the mutation operation.
        
        :return: None: 'enabled' hasn't changed
                 True: 'enabled' is set to True
                 False: 'enabled' is set to False
        Return True if enable has mutated."""
        mut_enabled = None
        for a in self._attributes:
            v = getattr(self, a.name)
            if a.name == 'enabled':
                v2 = a.mutate_value(v, config)
                if v != v2:
                    setattr(self, a.name, v2)
                    mut_enabled = v2
            else:
                setattr(self, a.name, a.mutate_value(v, config))
        return mut_enabled
