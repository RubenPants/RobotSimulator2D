"""
config.py

Class containing all the used configurations.
"""
from utils.dictionary import *


class GameConfig:
    """Default configuration for the games."""
    
    __slots__ = (
        "bot_driving_speed", "bot_radius", "bot_turning_speed",
        "batch", "duration", "max_game_id", "max_eval_game_id", "fps",
        "p2m", "x_axis", "y_axis",
        "noise_time", "noise_angle", "noise_distance", "noise_proximity",
        "sensor_ray_distance",
        "target_reached"
    )
    
    def __init__(self):
        # [BOT]
        # Maximal driving speed (driving straight) of the robot expressed in m/s  [def=0.6]
        self.bot_driving_speed: float = 0.6
        # Radius of the bot expressed in meters  [def=0.085]
        self.bot_radius: float = 0.085
        # Maximal turning speed of the robot expressed in radians per second  [def=3.53~=0.6/0.17]
        self.bot_turning_speed: float = 3.53
        
        # [CONTROL]
        # Number of games on which a single genome is evaluated  [def=12]  TODO
        self.batch: int = 10
        # Number of seconds it takes for one game to complete  [def=100]  TODO
        self.duration: int = 40
        # Max ID of game (starting from 1)  [def=1000]
        self.max_game_id: int = 1000
        # Max ID of evaluation game (starting from max_id)  [def=1100]
        self.max_eval_game_id: int = 1100
        # Number of frames each second  [def=20]
        self.fps: int = 20
        
        # [CREATION]
        # Pixel-to-meters: number of pixels that represent one meter  [def=50]
        self.p2m: int = 50
        # Number of meters the x-axis represents  [def=14]
        self.x_axis: int = 14
        # Number of meters the y-axis represents  [def=14]
        self.y_axis: int = 14
        
        # [NOISE]
        # Alpha in Gaussian distribution for time noise, max 0.02s deviation  [def=0.005]
        self.noise_time: float = 0.005
        # Alpha in Gaussian distribution for angular sensor, max 0.7° deviation  [def=0.001]
        self.noise_angle: float = 0.001
        # Alpha in Gaussian distribution for distance sensor, max ~1.5cm deviation  [def=0.005]
        self.noise_distance: float = 0.005
        # Alpha in Gaussian distribution for proximity sensor, max ~1.5cm deviation  [def=0.005]
        self.noise_proximity: float = 0.005
        
        # [SENSOR]
        # Distance a ray-sensor reaches, expressed in meters  [def=1.0]
        self.sensor_ray_distance: float = 1.0
        
        # [TARGET]
        # Target is reached when within this range, expressed in meters  [def=0.5]
        self.target_reached: float = 0.5
    
    def __str__(self):
        result = "Game configuration:"
        for k in self.__slots__:
            result += f"\n\t- {k} = {self.__getattribute__(k)}"
        return result


class NeatConfig:
    """Default configuration for the population."""
    
    __annotations__ = {  # Not as it should, but I'll do it anyways
        'NEAT':                ["fitness_criterion", "fitness_threshold", "no_fitness_termination", "pop_size"],
        'DefaultReproduction': ['elitism', 'min_species_size', 'parent_selection', 'sexual_reproduction'],
        'DefaultGenome':       ['activation_default', 'activation_mutate_rate', 'activation_options',
                                'aggregation_default', 'aggregation_mutate_rate', 'aggregation_options',
                                'bias_init_mean', 'bias_init_stdev', 'bias_max_value', 'bias_min_value',
                                'bias_mutate_power', 'bias_mutate_rate', 'bias_replace_rate',
                                'compatibility_disjoint_coefficient', 'compatibility_weight_coefficient',
                                'conn_add_prob', 'conn_delete_prob', 'enabled_default', 'enabled_mutate_rate',
                                'gru_enabled', 'gru_init_mean', 'gru_init_stdev', 'gru_max_value', 'gru_min_value',
                                'gru_mutate_power', 'gru_mutate_rate', 'gru_node_prob', 'gru_replace_rate',
                                'initial_connection', 'node_add_prob', 'node_delete_prob', 'num_hidden', 'num_outputs',
                                'weight_init_mean', 'weight_init_stdev', 'weight_max_value', 'weight_min_value',
                                'weight_mutate_power', 'weight_mutate_rate', 'weight_replace_rate'],
        'DefaultSpecies':      ['compatibility_threshold', 'max_stagnation', 'species_elitism', 'species_fitness_func',
                                'species_max', 'specie_stagnation'],
        'Evaluation':          ["fitness", "fitness_comb", "nn_k"],
    }
    
    def __init__(self):
        # [NEAT]
        # The function used to compute the termination criterion from the set of genome fitness [def=D_MAX]
        self.fitness_criterion: str = D_MAX
        # When fitness computed by fitness_criterion meets this threshold, the evolution process will terminate  [def=1]
        self.fitness_threshold: int = 1
        # Don't consider fitness_criterion and fitness_threshold  [def=True]
        self.no_fitness_termination: bool = True
        # Number of individuals in each generation  [def=128]  TODO
        self.pop_size: int = 256
        
        # [DefaultReproduction]
        # Number of most fit individuals per specie that are preserved as-is from one generation to the next  [def=3]
        self.elitism: int = 3
        # The fraction for each species allowed to reproduce each generation (parent selection)  [def=0.3]  TODO
        self.parent_selection: float = 0.1
        # Minimum number of genomes per species, keeping low prevents number of individuals blowing up  [def=10]  TODO
        self.min_species_size: int = 10
        # Sexual reproduction  [def=True]
        self.sexual_reproduction: bool = False
        
        # [DefaultGenome]
        # Initial node activation function  [def=D_GELU]
        self.activation_default: str = D_GELU
        # Probability of changing the activation function  [def=0]
        self.activation_mutate_rate: float = 0.0
        # All possible activation functions between whom can be switched during mutation  [def=D_GELU]
        self.activation_options: str = D_GELU
        # The default aggregation function attribute assigned to new nodes  [def=D_SUM]
        self.aggregation_default: str = D_SUM
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
        # Enable the algorithm to disable (and re-enable) existing connections  [def=True]
        self.enabled_default: bool = True
        # The probability that mutation will replace the 'enabled status' of a connection  [def=0.05]
        self.enabled_mutate_rate: float = 0.05
        # Initial connectivity of newly-created genomes  [def=D_PARTIAL_DIRECT_05]  TODO
        self.initial_connection = D_FULL_NODIRECT
        # Probability of adding a node during mutation (each generation)  [def=0.01]  TODO
        self.node_add_prob: float = 0.01
        # Probability of removing a node during mutation (each generation)  [def=0.01]  TODO
        self.node_delete_prob: float = 0.01
        # Number of hidden nodes to add to each genome in the initial population  [def=0]  TODO
        self.num_hidden: int = 1
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
        self.weight_mutate_power: float = 0.1
        # Probability of a weight (connection) to mutate  [def=0.2]  TODO
        self.weight_mutate_rate: float = 0.2
        # Probability of assigning completely new value, based on weight_init_mean and weight_init_stdev  [def=0.05]
        self.weight_replace_rate: float = 0.05
        
        # [DefaultSpecies]
        # Individuals whose genetic distance is less than this threshold are in the same specie  [def=2.0]  TODO
        self.compatibility_threshold: float = 1.0
        # Remove a specie if it hasn't improved over this many number of generations  [def=15]
        self.max_stagnation: int = 15
        # Number of the best species that will be protected from stagnation  [def=2]  TODO
        self.species_elitism: int = 1
        # The function used to compute the species fitness  [def=D_MAX]
        self.species_fitness_func: str = D_MAX
        # Maximum number of species that can live along each other  [def=10]
        self.species_max: int = 15
        # Number of generations before a previous elite specie can become stagnant  [def=5]
        self.specie_stagnation: int = 5
        
        # [EVALUATION]
        # Fitness functions [distance, diversity, novelty, path]  TODO
        self.fitness: str = D_DISTANCE
        # Function to combine the fitness-values across different games, choices are: min, avg, max, gmean  [def=gmean]
        self.fitness_comb: str = D_GMEAN
        # Number of nearest neighbors taken into account for a NN-utilizing fitness function  [def=3]
        self.nn_k: int = 3
        # Safe zone during novelty search, neighbours outside this range are not taken into account  [def=1]
        self.safe_zone: float = 1
        
        # [GRU]
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
        # The standard deviation of the zero-centered gaussian from which a GRU value mutation is drawn  [def=0.1]
        self.gru_mutate_power: float = 0.1
        # Probability of a GRU value to mutate  [def=0.2]  TODO
        self.gru_mutate_rate: float = 0.2
        # Probability of mutating a GRU node rather than a simple node  [def=0.6]  TODO
        self.gru_node_prob: float = 0.6
        # Probability of assigning completely new value, based on gru_init_mean and gru_init_stdev  [def=0.05]
        self.gru_replace_rate: float = 0.01
        
        # [SelfAdaptive]  TODO
    
    def __str__(self):
        result = "NEAT Configuration:"
        for k, v in self.__dict__.items():
            result += f"\n\t- {k} = {v}"
        return result
