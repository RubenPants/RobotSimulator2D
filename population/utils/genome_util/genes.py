"""
genes.py

Handles node and connection genes.
"""
from abc import abstractmethod
from copy import deepcopy

from numpy import concatenate as cat, zeros
from numpy.linalg import norm
from torch import float64, tensor
from torch.nn import GRUCell

from configs.genome_config import GenomeConfig
from population.utils.attributes import activation, aggregation, bias, conn_enabled, conn_weight, gru
from utils.dictionary import D_TANH


class BaseGene(object):
    """BaseGene specifies which methods a gene-object must implement."""
    
    __slots__ = {
        'key',
    }
    
    def __init__(self, key):
        """Key to identify the gene. Each gene has a distinct key."""
        self.key = key
    
    @abstractmethod
    def __str__(self):
        raise NotImplementedError(f"__str__ not set for gene with key {self.key}")
    
    @abstractmethod
    def __repr__(self):
        raise NotImplementedError(f"__repr__ not set for gene with key {self.key}")
    
    def __lt__(self, other):
        assert isinstance(self.key, type(other.key)), f"Cannot compare keys {self.key!r} and {other.key!r}"
        return self.key < other.key
    
    def copy(self, cfg):
        new_gene = self.__class__(self.key, cfg=cfg)
        for param in self.__slots__:
            attr = getattr(self, param)
            
            # Data-types that carry no references
            if type(attr) in [int, float, bool, str, complex]:
                setattr(new_gene, param, attr)
            
            # Deepcopy the attributes that carry references
            else:
                setattr(new_gene, param, deepcopy(attr))
        return new_gene
    
    @abstractmethod
    def crossover(self, cfg: GenomeConfig, other, ratio: float = 0.5):
        """Create a new gene based on the current configuration and that of another (same-class) gene."""
        raise NotImplementedError(f"Crossover is not implemented for gene {self.key}")
    
    @abstractmethod
    def distance(self, other, cfg: GenomeConfig):
        raise NotImplementedError(f"Distance is not implemented for gene {self.key}")
    
    @abstractmethod
    def mutate(self, cfg: GenomeConfig):
        raise NotImplementedError(f"Mutation is not implemented for gene {self.key}")


class SimpleNodeGene(BaseGene):
    """Simple node configuration, as specified by the Python-NEAT documentation."""
    
    __slots__ = {
        'bias', 'activation', 'aggregation',
    }
    
    def __init__(self, key, cfg: GenomeConfig):
        assert isinstance(key, int), f"SimpleNodeGene key must be an int, not {key!r}"
        BaseGene.__init__(self, key)
        
        # Initialize gene attributes
        self.activation: str = activation.init(cfg)
        self.aggregation: str = aggregation.init(cfg)
        self.bias: float = bias.init(cfg)
    
    def __str__(self):
        return f"SimpleNodeGene(\n" \
               f"\tactivation={self.activation}\n" \
               f"\taggregation={self.aggregation},\n" \
               f"\tbias={round(self.bias, 2)})"
    
    def __repr__(self):
        return f"SimpleNodeGene(bias={round(self.bias, 2)})"
    
    def crossover(self, cfg: GenomeConfig, other, ratio: float = 0.5):
        assert self.__class__ == other.__class__ == SimpleNodeGene
        assert self.key == other.key
        new_gene = SimpleNodeGene(self.key, cfg=cfg)
        new_gene.activation = activation.cross(v1=self.activation, v2=other.activation, ratio=ratio)
        new_gene.aggregation = aggregation.cross(v1=self.aggregation, v2=other.aggregation, ratio=ratio)
        new_gene.bias = bias.cross(v1=self.bias, v2=other.bias, ratio=ratio)
        return new_gene
    
    def distance(self, other, cfg: GenomeConfig):
        d = abs(self.bias - other.bias)
        if self.activation != other.activation: d += 1.0
        if self.aggregation != other.aggregation: d += 1.0  # Normally, aggregation is always equal to 'sum'
        return d * cfg.compatibility_weight_coefficient
    
    def mutate(self, cfg: GenomeConfig):
        self.activation = activation.mutate(self.activation, cfg=cfg)
        self.aggregation = aggregation.mutate(self.aggregation, cfg=cfg)
        self.bias = bias.mutate(self.bias, cfg=cfg)


class OutputNodeGene(BaseGene):
    """Node representation for each of the network's outputs."""
    
    __slots__ = {
        'activation', 'aggregation', 'bias',
    }
    
    def __init__(self, key, cfg: GenomeConfig):
        assert isinstance(key, int), f"OutputNodeGene key must be an int, not {key!r}"
        BaseGene.__init__(self, key)
        
        # Initialize gene attributes
        self.activation: str = D_TANH
        self.aggregation: str = aggregation.init(cfg)
        self.bias: float = bias.init(cfg)
    
    def __str__(self):
        return f"OutputNodeGene(\n" \
               f"\tactivation={self.activation}\n" \
               f"\taggregation={self.aggregation},\n" \
               f"bias={round(self.bias, 2)})"
    
    def __repr__(self):
        return f"OutputNodeGene(bias={round(self.bias, 2)})"
    
    def crossover(self, cfg: GenomeConfig, other, ratio: float = 0.5):
        assert self.__class__ == other.__class__ == OutputNodeGene
        assert self.key == other.key
        new_gene = OutputNodeGene(self.key, cfg=cfg)
        new_gene.aggregation = aggregation.cross(v1=self.aggregation, v2=other.aggregation, ratio=ratio)
        new_gene.bias = bias.cross(v1=self.bias, v2=other.bias, ratio=ratio)
        return new_gene
    
    def distance(self, other, cfg: GenomeConfig):
        """Only possible difference in output-nodes' distance is the bias."""
        return abs(self.bias - other.bias) * cfg.compatibility_weight_coefficient
    
    def mutate(self, cfg: GenomeConfig):
        """Mutation is not possible on the activation."""
        self.aggregation = aggregation.mutate(self.aggregation, cfg=cfg)
        self.bias = bias.mutate(self.bias, cfg=cfg)


class GruNodeGene(BaseGene):
    """Custom GRU cell implementation."""
    
    __slots__ = {
        'activation', 'bias', 'bias_hh', 'bias_ih', 'input_keys', 'input_keys_full', 'weight_hh', 'weight_ih',
        'weight_ih_full'
    }
    
    def __init__(self, key, cfg: GenomeConfig, input_keys=None, input_keys_full=None):
        assert isinstance(key, int), f"OutputNodeGene key must be an int, not {key!r}"
        if input_keys and input_keys_full:
            for k in input_keys: assert k in input_keys_full
        BaseGene.__init__(self, key)
        
        # Initialize gene attributes
        self.activation = activation.init(cfg)
        self.bias = 0  # Needed for feed-forward-network
        self.bias_hh = gru.init(cfg)
        self.bias_ih = gru.init(cfg)
        self.input_keys: list = input_keys if input_keys else []
        self.input_keys_full: list = input_keys_full if input_keys_full else []
        self.weight_hh = gru.init(cfg, input_size=1)
        self.weight_ih = None  # Updated via update_weight_ih
        self.weight_ih_full = gru.init(cfg, input_size=len(self.input_keys_full))
        
        # Make sure that the GRU-cell is initialized correct
        self.update_weight_ih()
    
    def __str__(self):
        return f"GruNodeGene(\n" \
               f"\tbias_hh={[round(float(x), 2) for x in self.bias_hh]},\n" \
               f"\tbias_ih={[round(float(x), 2) for x in self.bias_ih]},\n" \
               f"\tinput_keys={self.input_keys},\n" \
               f"\tweight_hh={[round(float(x), 2) for x in self.weight_hh]},\n" \
               f"\tweight_ih={[round(float(x), 2) for x in self.weight_ih]})"
    
    def __repr__(self):
        return f"GruNodeGene(inputs={self.input_keys!r})"
    
    def crossover(self, cfg: GenomeConfig, other, ratio: float = 0.5):
        assert self.__class__ == other.__class__ == GruNodeGene
        assert self.key == other.key
        
        # Initialize a randomized gene
        input_keys_full = sorted(set(self.input_keys_full + other.input_keys_full))
        new_gene = GruNodeGene(self.key, cfg=cfg, input_keys=[], input_keys_full=input_keys_full)
        assert new_gene.input_keys == []
        assert new_gene.input_keys_full == input_keys_full
        assert new_gene.weight_ih_full.shape == (3, len(input_keys_full))
        
        # Crossover the weight_ih_full attribute
        for i, k in enumerate(new_gene.input_keys_full):
            if k in self.input_keys_full and k in other.input_keys_full:  # Key is shared by both
                new_gene.weight_ih_full[:, i] = gru.cross_1d(
                        v1=self.weight_ih_full[:, self.input_keys_full.index(k)],
                        v2=other.weight_ih_full[:, other.input_keys_full.index(k)],
                        ratio=ratio,
                )
            elif k in self.input_keys_full:  # Key only contained by first parent (self)
                new_gene.weight_ih_full[:, i] = self.weight_ih_full[:, self.input_keys_full.index(k)]
            else:  # Key only contained by second parent (other)
                new_gene.weight_ih_full[:, i] = other.weight_ih_full[:, other.input_keys_full.index(k)]
        
        # Crossover the other attributes
        new_gene.bias_hh = gru.cross_1d(self.bias_hh, other.bias_hh, ratio=ratio)
        new_gene.bias_ih = gru.cross_1d(self.bias_ih, other.bias_ih, ratio=ratio)
        new_gene.weight_hh = gru.cross_2d(self.weight_hh, other.weight_hh, ratio=ratio)
        return new_gene
    
    def distance(self, other, cfg: GenomeConfig):
        """Calculate the average distance between two GRU nodes, which is determined by its coefficients."""
        d = 0
        d += norm(self.bias_ih - other.bias_ih)
        d += norm(self.bias_hh - other.bias_hh)
        d += norm(self.weight_hh - other.weight_hh)
        
        # Compare only same input keys
        key_set = sorted(set(self.input_keys + other.input_keys))
        s = zeros((3, len(key_set)), dtype=float)
        o = zeros((3, len(key_set)), dtype=float)
        for i, k in enumerate(key_set):
            if k in self.input_keys: s[:, i] = self.weight_ih_full[:, self.input_keys_full.index(k)]
            if k in other.input_keys: o[:, i] = other.weight_ih_full[:, other.input_keys_full.index(k)]
        d += norm(s - o)
        if self.activation != other.activation: d += 1.0
        return (d / 4) * cfg.compatibility_weight_coefficient
    
    def mutate(self, cfg: GenomeConfig):
        self.activation = activation.mutate(self.activation, cfg=cfg)
        self.bias_hh = gru.mutate_1d(self.bias_hh, cfg=cfg)
        self.bias_ih = gru.mutate_1d(self.bias_ih, cfg=cfg)
        self.weight_hh = gru.mutate_2d(self.weight_hh, cfg=cfg)
        self.weight_ih_full = gru.mutate_2d(self.weight_ih_full, cfg=cfg,
                                            mapping=[k in self.input_keys for k in self.input_keys_full])
    
    def update_weight_ih(self):
        """Update weight_ih to be conform with the current input_keys-set."""
        self.weight_ih = zeros((3, len(self.input_keys)), dtype=float)
        for i, k in enumerate(self.input_keys):
            self.weight_ih[:, i] = self.weight_ih_full[:, self.input_keys_full.index(k)]
    
    def get_gru(self, mapping=None):  # TODO: Document weight_map
        """Return a PyTorch GRUCell based on current configuration. The mapping denotes which columns to use."""
        self.update_weight_ih()
        if mapping is not None:
            cell = GRUCell(input_size=len(mapping[mapping]), hidden_size=1)
            cell.weight_ih[:] = tensor(self.weight_ih[:, mapping], dtype=float64)
        else:
            cell = GRUCell(input_size=len(self.input_keys), hidden_size=1)
            cell.weight_ih[:] = tensor(self.weight_ih, dtype=float64)
        cell.weight_hh[:] = tensor(self.weight_hh, dtype=float64)
        cell.bias_ih[:] = tensor(self.bias_ih, dtype=float64)
        cell.bias_hh[:] = tensor(self.bias_hh, dtype=float64)
        return cell
    
    def add_input_key(self, cfg: GenomeConfig, k: int):
        """Extend the input-key list with the given key, and expand the corresponding weights."""
        # Update self.weight_ih_full if key never seen before
        if k not in self.input_keys_full:
            # Find the index to insert the key
            lst = [i + 1 for i in range(len(self.input_keys_full)) if self.input_keys_full[i] < k]  # List of indices
            i = lst[-1] if lst else 0  # Index to insert key in
            
            # Save key to list
            self.input_keys_full.insert(i, k)
            
            # Update weight_ih_full correspondingly by inserting random initialized tensor in correct position
            new_tensor = gru.init(cfg, input_size=1)
            assert new_tensor.shape == (3, 1)
            self.weight_ih_full = cat((self.weight_ih_full[:, :i], new_tensor, self.weight_ih_full[:, i:]), axis=1)
        
        # Update input_keys (current key-set) analogously
        if k not in self.input_keys:
            lst = [i + 1 for i in range(len(self.input_keys)) if self.input_keys[i] < k]  # List of indices
            i = lst[-1] if lst else 0
            self.input_keys.insert(i, k)
    
    def remove_input_key(self, k):
        """Delete one of the input_keys, input_keys_full and weight_ih_full are left unchanged."""
        if k in self.input_keys: self.input_keys.remove(k)


class ConnectionGene(BaseGene):
    """Connection configuration, as specified by the Python-NEAT documentation."""
    
    __slots__ = {
        'enabled', 'weight',
    }
    
    def __init__(self, key, cfg: GenomeConfig):
        assert isinstance(key, tuple), f"ConnectionGene key must be a tuple, not {key!r}"
        BaseGene.__init__(self, key)
        
        # Initialize gene attributes
        self.enabled = conn_enabled.init(cfg)
        self.weight = conn_weight.init(cfg)
    
    def __str__(self):
        return f"ConnectionGene(\n" \
               f"\tenabled={self.enabled}\n" \
               f"\tweight={round(self.weight, 2)})"
    
    def __repr__(self):
        return f"ConnectionGene(weight={round(self.weight, 2)}, enabled={self.enabled})"
    
    def crossover(self, cfg: GenomeConfig, other, ratio: float = 0.5):
        assert self.__class__ == other.__class__ == ConnectionGene
        assert self.key == other.key
        new_gene = ConnectionGene(self.key, cfg=cfg)
        new_gene.enabled = conn_enabled.cross(v1=self.enabled, v2=other.enabled, ratio=ratio)
        new_gene.weight = conn_weight.cross(v1=self.weight, v2=other.weight, ratio=ratio)
        return new_gene
    
    def distance(self, other, config):
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled: d += 1.0
        return d * config.compatibility_weight_coefficient
    
    def mutate(self, cfg: GenomeConfig):
        """
        Mutate the connection's attributes. Mutate the enabled state as last since it returns a value.
        
        :param cfg: GenomeConfig object
        :return: None: 'enabled' hasn't changed
                 True: 'enabled' is set to True
                 False: 'enabled' is set to False
        """
        self.weight = conn_weight.mutate(self.weight, cfg=cfg)
        pre = self.enabled
        self.enabled = conn_enabled.mutate(self.enabled, cfg=cfg)
        if pre != self.enabled:
            return self.enabled
        return None
