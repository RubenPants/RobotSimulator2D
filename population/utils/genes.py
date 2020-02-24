"""
genes.py

Handles node and connection genes.
"""
import warnings
from random import random

from neat.attributes import BoolAttribute, FloatAttribute, StringAttribute

from utils.dictionary import D_TANH, D_ACTIVATION


class BaseGene(object):
    """
    Handles functions shared by multiple types of genes (both node and connection), including crossover and calling
    mutation methods.
    """
    
    _gene_attributes = None
    
    def __init__(self, key):
        self.key = key
    
    def __str__(self):
        attrib = ['key'] + [a.name for a in self._gene_attributes]
        attrib = ['{0}={1}'.format(a, getattr(self, a)) for a in attrib]
        return '{0}({1})'.format(self.__class__.__name__, ", ".join(attrib))
    
    def __lt__(self, other):
        assert isinstance(self.key, type(other.key)), f"Cannot compare keys {self.key!r} and {other.key!r}"
        return self.key < other.key
    
    @classmethod
    def parse_config(cls, config, param_dict):
        pass
    
    @classmethod
    def get_config_params(cls):
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
                        FloatAttribute('response'),
                        StringAttribute('activation', options='tanh'),
                        StringAttribute('aggregation', options='sum')]
    
    def __init__(self, key):
        # Placeholders
        self.response = None
        self.bias = None
        self.activation = None
        self.aggregation = None
        
        assert isinstance(key, int), "DefaultNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)
    
    def distance(self, other, config):
        d = abs(self.bias - other.bias) + abs(self.response - other.response)
        # if self.activation != other.activation: d += 1.0  TODO: Take difference in activation into account?
        if self.aggregation != other.aggregation: d += 1.0
        return d * config.compatibility_weight_coefficient


class OutputNodeGene(DefaultNodeGene):
    def __init__(self, key):
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
            if a.name != D_ACTIVATION:
                setattr(self, a.name, a.mutate_value(v, config))


class DefaultConnectionGene(BaseGene):
    """Default connection configuration, as specified by the Python-NEAT documentation."""
    
    _gene_attributes = [FloatAttribute('weight'),
                        BoolAttribute('enabled')]
    
    def __init__(self, key):
        # Placeholders
        self.enabled = None
        self.weight = None
        
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)
    
    def distance(self, other, config):
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient
