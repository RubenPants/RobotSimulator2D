"""
attributes.py

TODO: Delete this file!
 It's ridiculous to create new objects each mutation! Make separate methods for it. (new attribute-specific folder)

Deals with the attributes (variable parameters) of genes.
"""
from random import choice, gauss, random

import numpy as np
import torch
from neat.config import ConfigParameter
from neat.six_util import iteritems, iterkeys


class BaseAttribute(object):
    """Superclass for the type-specialized attribute subclasses, used by genes."""
    
    _config_items = None  # Placeholder
    
    def __init__(self, name, **default_dict):
        self.name = name
        for n, default in iteritems(default_dict): self._config_items[n] = [self._config_items[n][0], default]
        for n in iterkeys(self._config_items): setattr(self, n + "_name", self.config_item_name(n))
    
    def config_item_name(self, config_item_base_name):
        return f"{self.name}_{config_item_base_name}"
    
    def get_config_params(self):
        return [ConfigParameter(self.config_item_name(n), self._config_items[n][0], self._config_items[n][1])
                for n in iterkeys(self._config_items)]


class FloatAttribute(BaseAttribute):
    """Class for numeric attributes, such as the bias of a node or the weight of a connection."""
    
    _config_items = {"init_mean":    [float, None],
                     "init_stdev":   [float, None],
                     "init_type":    [str, 'gaussian'],
                     "replace_rate": [float, None],
                     "mutate_rate":  [float, None],
                     "mutate_power": [float, None],
                     "max_value":    [float, None],
                     "min_value":    [float, None]}
    
    def __init__(self, name, **default_dict):
        self.init_mean_name = None  # Placeholder
        self.init_stdev_name = None  # Placeholder
        self.min_value_name = None  # Placeholder
        self.max_value_name = None  # Placeholder
        self.init_type_name = None  # Placeholder
        self.mutate_rate_name = None  # Placeholder
        self.mutate_power_name = None  # Placeholder
        self.replace_rate_name = None  # Placeholder
        super().__init__(name, **default_dict)
    
    def clamp(self, value, config):
        return max(min(value, getattr(config, self.max_value_name)), getattr(config, self.min_value_name))
    
    def init_value(self, config):
        mean = getattr(config, self.init_mean_name)
        stdev = getattr(config, self.init_stdev_name)
        return self.clamp(gauss(mean, stdev), config)
    
    def mutate_value(self, value, config):
        # mutate_rate is usually no lower than replace_rate, and frequently higher - so put first for efficiency
        mutate_rate = getattr(config, self.mutate_rate_name)
        
        r = random()
        if r < mutate_rate:
            mutate_power = getattr(config, self.mutate_power_name)
            return self.clamp(value + gauss(0.0, mutate_power), config)
        
        replace_rate = getattr(config, self.replace_rate_name)
        
        if r < replace_rate + mutate_rate: return self.init_value(config)
        return value
    
    def validate(self, config):  # pragma: no cover
        pass


class BoolAttribute(BaseAttribute):
    """Class for boolean attributes such as whether a connection is enabled or not."""
    
    _config_items = {"default":           [str, None],
                     "mutate_rate":       [float, None],}
    
    def __init__(self, name, **default_dict):
        self.default_name = None  # Placeholder
        self.mutate_rate_name = None  # Placeholder
        super().__init__(name, **default_dict)
    
    def init_value(self, config):
        default = str(getattr(config, self.default_name)).lower()
        
        if default in ('1', 'on', 'yes', 'true'):
            return True
        elif default in ('0', 'off', 'no', 'false'):
            return False
        elif default in ('random', 'none'):
            return bool(random() < 0.5)
        
        raise RuntimeError(f"Unknown default value {default!r} for {self.name!s}")
    
    def mutate_value(self, value, config):
        mutate_rate = getattr(config, self.mutate_rate_name)
        
        if mutate_rate > 0:
            r = random()
            # NOTE: we choose a random value here so that the mutation rate has the same exact meaning as the rates
            # given for the string and bool attributes (the mutation operation *may* change the value but is not
            # guaranteed to do so).
            if r < mutate_rate: return random() < 0.5
        return value
    
    def validate(self, config):  # pragma: no cover
        pass


class StringAttribute(BaseAttribute):
    """
    Class for string attributes such as the aggregation function of a node, which are selected from a list of options.
    """
    
    _config_items = {"default":     [str, 'random'],
                     "options":     [list, None],
                     "mutate_rate": [float, None]}
    
    def __init__(self, name, **default_dict):
        self.default_name = None  # Placeholder
        self.mutate_rate_name = None  # Placeholder
        self.options_name = None  # Placeholder
        super().__init__(name, **default_dict)
    
    def init_value(self, config):
        default = getattr(config, self.default_name)
        
        if default.lower() in ('none', 'random'):
            options = getattr(config, self.options_name)
            return choice(options)
        return default
    
    def mutate_value(self, value, config):
        mutate_rate = getattr(config, self.mutate_rate_name)
        
        if mutate_rate > 0:
            r = random()
            if r < mutate_rate:
                options = getattr(config, self.options_name)
                return choice(options)
        return value
    
    def validate(self, config):  # pragma: no cover
        pass


class GruBiasAttribute(object):
    """Bias attribute used in the GRU-gene. The bias can be seen as a vector of floats."""
    
    def __init__(self, name, **default_dict):
        self.name = name
        self.fa = FloatAttribute("gru", **default_dict)  # Init FloatAttribute 'fa' to perform float calculations on
    
    def init_value(self, config):
        """Create a vector with on each specified position a FloatAttribute."""
        tensor = torch.tensor(np.zeros((3,)), dtype=torch.float64)
        
        # Query the FloatAttribute for each initialization of the tensor's parameters
        for t_index in range(len(tensor)): tensor[t_index] = self.fa.init_value(config=config)
        return tensor
    
    def mutate_value(self, value, config):
        """Mutate the FloatTensor."""
        # Query the FloatAttribute for each initialization of the tensor's parameters
        for t_index in range(len(value)):
            value[t_index] = self.fa.mutate_value(value=float(value[t_index]), config=config)
        return value
    
    def validate(self, config):  # pragma: no cover
        pass


class GruWeightAttribute(object):
    """
    Weight attribute used in the GRU-gene. The GRU-weight can be seen as a vector of floats, similar to the bias
    attribute.
    """
    
    def __init__(self, name, **default_dict):
        self.name = name
        self.fa = FloatAttribute("gru", **default_dict)  # Init FloatAttribute 'fa' to perform float calculations on
    
    def init_value(self, config, input_size=None):
        """Create a vector with on each specified position a FloatAttribute."""
        if input_size is None: input_size = 1
        tensor = torch.tensor(np.zeros((3, input_size)), dtype=torch.float64)
        
        # Query the FloatAttribute for each initialization of the tensor's parameters
        for x_index, y_index in np.ndindex(tensor.shape): tensor[x_index, y_index] = self.fa.init_value(config=config)
        return tensor
    
    def mutate_value(self, value, config, mapping=None):
        """Mutate the FloatTensor."""
        # Query the FloatAttribute for each initialization of the tensor's parameters
        for row, col in np.ndindex(value.shape):
            if mapping is not None and not mapping[col]: continue
            value[row, col] = self.fa.mutate_value(value=float(value[row, col]), config=config)
        return value
    
    def validate(self, config):  # pragma: no cover
        pass
