"""
population_config.py

Population configuration specific to the NEAT parameters. This file does general configuration parsing; used by other
classes for their configuration.
"""
from __future__ import print_function

import warnings

from neat.six_util import iterkeys

from config import NeatConfig
from population.utils.config.genome_config import DefaultGenomeConfig


class ConfigParameter(object):
    """Contains information about one configuration item."""
    
    def __init__(self, name, value_type, default=None):
        self.name = name
        self.value_type = value_type
        self.default = default
    
    def __repr__(self):
        if self.default is None: return f"ConfigParameter({self.name!r}, {self.value_type!r})"
        return f"ConfigParameter({self.name!r}, {self.value_type!r}, {self.default!r})"
    
    def parse(self, section, config_parser):
        if int == self.value_type: return config_parser.getint(section, self.name)
        if bool == self.value_type: return config_parser.getboolean(section, self.name)
        if float == self.value_type: return config_parser.getfloat(section, self.name)
        if list == self.value_type:
            v = config_parser.get(section, self.name)
            return v.split(" ")
        if str == self.value_type: return config_parser.get(section, self.name)
        raise RuntimeError(f"Unexpected configuration type: {repr(self.value_type)}")
    
    def interpret(self, config_dict):
        """
        Converts the config_parser output into the proper type,
        supplies defaults if available and needed, and checks for some errors.
        """
        value = config_dict.get(self.name)
        if value is None:
            if self.default is None:
                raise RuntimeError(f'Missing configuration item: {self.name}')
            else:
                warnings.warn(f"Using default {self.default!r} for '{self.name!s}'", DeprecationWarning)
                if (str != self.value_type) and isinstance(self.default, self.value_type):
                    return self.default
                else:
                    value = self.default
        
        try:
            if str == self.value_type: return str(value)
            if int == self.value_type: return int(value)
            if bool == self.value_type:
                if value.lower() == "true":
                    return True
                elif value.lower() == "false":
                    return False
                else:
                    raise RuntimeError(self.name + " must be True or False")
            if float == self.value_type: return float(value)
            if list == self.value_type: return value.split(" ")
        except Exception:
            raise RuntimeError(
                    f"Error interpreting config '{self.name}' with value {value!r} and type {self.value_type}")
        raise RuntimeError(f"Unexpected configuration type: {repr(self.value_type)}")
    
    def format(self, value):
        if list == self.value_type: return " ".join(value)
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
        # Placeholders
        self.elitism = None
        
        # Initialize the parameters
        self._params = param_list
        param_list_names = []
        for p in param_list:
            setattr(self, p.name, p.interpret(param_dict))
            param_list_names.append(p.name)
        unknown_list = [x for x in iterkeys(param_dict) if not x in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown configuration items:\n" + "\n\t".join(unknown_list))
            raise UnknownConfigItemError("Unknown configuration item {!s}".format(unknown_list[0]))
    
    def __str__(self, name=''):
        """Readable format for the configuration."""
        attrib = [a.name for a in self._params]
        result = f"Default {name + ' ' if name else ''}configuration:"
        for a in attrib:
            attr = getattr(self, a)
            if isinstance(attr, float):
                result += f"\n\t- {a} = {round(attr, 3)}"
            else:
                result += f"\n\t- {a} = {attr}"
        return result
    
    @classmethod
    def write_config(cls, f, config):
        write_pretty_params(f, config, config._params)


class Config(object):
    """A simple container for user-configurable parameters of NEAT."""
    
    __params = [ConfigParameter('pop_size', int),
                ConfigParameter('fitness_criterion', str),
                ConfigParameter('fitness_threshold', float),
                ConfigParameter('no_fitness_termination', bool)]
    
    def __init__(self,
                 genome_type,
                 reproduction_type,
                 species_type,
                 stagnation_type,
                 config: NeatConfig):
        # config: NeatConfig):
        # Check that the provided types have the required methods.
        assert hasattr(genome_type, 'parse_config')
        assert hasattr(reproduction_type, 'parse_config')
        assert hasattr(species_type, 'parse_config')
        assert hasattr(stagnation_type, 'parse_config')
        
        # Set all the relevant types
        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_type = species_type
        self.stagnation_type = stagnation_type
        self.config = config
        
        # Placeholders
        self.pop_size = None
        self.fitness_criterion = None
        self.fitness_threshold = None
        self.no_fitness_termination = None
        
        # Load in the parameters
        param_list_names = []
        for p in self.__params:
            setattr(self, p.name, getattr(self.config, p.name))
            param_list_names.append(p.name)
        
        # Parse type sections:
        #  - Filter out the wanted parameters from the configs
        #  - Create a dictionary mapping the wanted parameters to their values (str format)
        #  - Initialize the needed entities
        genome_params = self.config.DefaultGenome
        genome_config = {param: str(getattr(self.config, param)) for param in genome_params}
        self.genome_config: DefaultGenomeConfig = genome_type.parse_config(genome_config)
        
        specie_params = self.config.DefaultSpecies
        species_config = {param: str(getattr(self.config, param)) for param in specie_params}
        self.species_config: DefaultClassConfig = species_type.parse_config(species_config)
        self.stagnation_config: DefaultClassConfig = stagnation_type.parse_config(species_config)  # TODO: Merge?
        
        reproduction_params = self.config.DefaultReproduction
        reproduction_config = {param: str(getattr(self.config, param)) for param in reproduction_params}
        self.reproduction_config: DefaultClassConfig = reproduction_type.parse_config(reproduction_config)
    
    def __str__(self):
        """Readable configuration file."""
        result = "Population configuration:"
        # Add genome configuration
        result += "\n\t"
        result += str(self.genome_config).replace("\t", "\t\t")
        # Add specie configuration
        result += "\n\n\t"
        result += self.species_config.__str__(name='species').replace("\t", "\t\t")
        # Add stagnation configuration
        result += "\n\n\t"
        result += self.stagnation_config.__str__(name='stagnation').replace("\t", "\t\t")
        # Add reproduction configuration
        result += "\n\n\t"
        result += self.reproduction_config.__str__(name='reproduction').replace("\t", "\t\t")
        return result
