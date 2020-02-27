"""
population_config.py

Population configuration specific to the NEAT parameters. This file does general configuration parsing; used by other
classes for their configuration.
"""
from __future__ import print_function

import warnings

from neat.six_util import iterkeys

from configs.config import NeatConfig
from utils.dictionary import D_ENABLE_GRU


class ConfigParameter(object):
    """Contains information about one configuration item."""
    
    def __init__(self, name, value_type, default=None):
        self.name = name
        self.value_type = value_type
        self.default = default
    
    def __repr__(self):
        if self.default is None:
            return "ConfigParameter({!r}, {!r})".format(self.name,
                                                        self.value_type)
        return "ConfigParameter({!r}, {!r}, {!r})".format(self.name,
                                                          self.value_type,
                                                          self.default)
    
    def parse(self, section, config_parser):
        if int == self.value_type:
            return config_parser.getint(section, self.name)
        if bool == self.value_type:
            return config_parser.getboolean(section, self.name)
        if float == self.value_type:
            return config_parser.getfloat(section, self.name)
        if list == self.value_type:
            v = config_parser.get(section, self.name)
            return v.split(" ")
        if str == self.value_type:
            return config_parser.get(section, self.name)
        
        raise RuntimeError("Unexpected configuration type: "
                           + repr(self.value_type))
    
    def interpret(self, config_dict):
        """
        Converts the config_parser output into the proper type,
        supplies defaults if available and needed, and checks for some errors.
        """
        value = config_dict.get(self.name)
        if value is None:
            if self.default is None:
                raise RuntimeError('Missing configuration item: ' + self.name)
            else:
                warnings.warn("Using default {!r} for '{!s}'".format(self.default, self.name),
                              DeprecationWarning)
                if (str != self.value_type) and isinstance(self.default, self.value_type):
                    return self.default
                else:
                    value = self.default
        
        try:
            if str == self.value_type:
                return str(value)
            if int == self.value_type:
                return int(value)
            if bool == self.value_type:
                if value.lower() == "true":
                    return True
                elif value.lower() == "false":
                    return False
                else:
                    raise RuntimeError(self.name + " must be True or False")
            if float == self.value_type:
                return float(value)
            if list == self.value_type:
                return value.split(" ")
        except Exception:
            raise RuntimeError("Error interpreting config item '{}' with value {!r} and type {}".format(
                    self.name, value, self.value_type))
        
        raise RuntimeError("Unexpected configuration type: " + repr(self.value_type))
    
    def format(self, value):
        if list == self.value_type:
            return " ".join(value)
        return str(value)


def write_pretty_params(f, config, params):
    param_names = [p.name for p in params]
    longest_name = max(len(name) for name in param_names)
    param_names.sort()
    params = dict((p.name, p) for p in params)
    
    for name in param_names:
        p = params[name]
        f.write('{} = {}\n'.format(p.name.ljust(longest_name), p.format(getattr(config, p.name))))


class UnknownConfigItemError(NameError):
    """Error for unknown configuration option - partially to catch typos."""
    pass


class DefaultClassConfig(object):
    """
    Replaces at least some boilerplate configuration code
    for reproduction, species_set, and stagnation classes.
    """
    
    def __init__(self, param_dict, param_list):
        self._params = param_list
        param_list_names = []
        for p in param_list:
            setattr(self, p.name, p.interpret(param_dict))
            param_list_names.append(p.name)
        unknown_list = [x for x in iterkeys(param_dict) if not x in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown configuration items:\n" +
                                             "\n\t".join(unknown_list))
            raise UnknownConfigItemError("Unknown configuration item {!s}".format(unknown_list[0]))
    
    @classmethod
    def write_config(cls, f, config):
        # pylint: disable=protected-access
        write_pretty_params(f, config, config._params)


class PopulationConfig(object):
    """A simple container for user-configurable parameters of NEAT."""
    
    __params = [ConfigParameter('pop_size', int),
                ConfigParameter('fitness_criterion', str),
                ConfigParameter('fitness_threshold', float),
                ConfigParameter('reset_on_extinction', bool),
                ConfigParameter('no_fitness_termination', bool)]
    
    def __init__(self,
                 genome_type,
                 reproduction_type,
                 species_set_type,
                 stagnation_type,
                 config: NeatConfig):
        # config: NeatConfig):
        # Check that the provided types have the required methods.
        assert hasattr(genome_type, 'parse_config')
        assert hasattr(reproduction_type, 'parse_config')
        assert hasattr(species_set_type, 'parse_config')
        assert hasattr(stagnation_type, 'parse_config')
        
        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type
        self.config = config
        
        param_list_names = []
        for p in self.__params:
            setattr(self, p.name, self.config.__dict__[p.name])
            param_list_names.append(p.name)
        
        # Parse type sections:
        #  - Filter out the wanted parameters from the configs
        #  - Create a dictionary mapping the wanted parameters to their values (str format)
        #  - Initialize the needed entities
        genome_params = self.config.__annotations__[genome_type.__name__] + [D_ENABLE_GRU]
        genome_dict = {k: str(v) for k, v in self.config.__dict__.items() if k in genome_params}
        self.genome_config = genome_type.parse_config(genome_dict)
        
        specie_params = self.config.__annotations__[species_set_type.__name__]
        species_set_dict = {k: str(v) for k, v in self.config.__dict__.items() if k in specie_params}
        self.species_set_config = species_set_type.parse_config(species_set_dict)
        
        stagnation_params = self.config.__annotations__[stagnation_type.__name__]
        stagnation_dict = {k: str(v) for k, v in self.config.__dict__.items() if k in stagnation_params}
        self.stagnation_config = stagnation_type.parse_config(stagnation_dict)
        
        reproduction_params = self.config.__annotations__[reproduction_type.__name__]
        reproduction_dict = {k: str(v) for k, v in self.config.__dict__.items() if k in reproduction_params}
        self.reproduction_config = reproduction_type.parse_config(reproduction_dict)
    
    def save(self, filename):
        with open(filename, 'w') as f:
            f.write('# The `NEAT` section specifies parameters particular to the NEAT algorithm\n')
            f.write('# or the experiment itself.  This is the only required section.\n')
            f.write('[NEAT]\n')
            write_pretty_params(f, self, self.__params)
            
            f.write('\n[{0}]\n'.format(self.genome_type.__name__))
            self.genome_type.write_config(f, self.genome_config)
            
            f.write('\n[{0}]\n'.format(self.species_set_type.__name__))
            self.species_set_type.write_config(f, self.species_set_config)
            
            f.write('\n[{0}]\n'.format(self.stagnation_type.__name__))
            self.stagnation_type.write_config(f, self.stagnation_config)
            
            f.write('\n[{0}]\n'.format(self.reproduction_type.__name__))
            self.reproduction_type.write_config(f, self.reproduction_config)
