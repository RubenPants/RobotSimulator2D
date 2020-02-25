import neat

from configs.config import NeatConfig
from population.utils.genome import DefaultGenome
from population.utils.population_config import PopulationConfig
from population.utils.reproduction import DefaultReproduction
from population.utils.species import DefaultSpeciesSet

cfg = NeatConfig()
config = PopulationConfig(
        genome_type=DefaultGenome,
        reproduction_type=DefaultReproduction,
        species_set_type=DefaultSpeciesSet,
        stagnation_type=neat.DefaultStagnation,
        config=cfg,
)

# Setup the genome-config
genome_dict = {k: str(v) for k, v in cfg.__dict__.items() if k in cfg.__annotations__[DefaultGenome.__name__]}
genome_config = DefaultGenome.parse_config(genome_dict)


def create_simple_genome():
    """A simple genome has one input, one output, and one hidden recurrent node."""
    genome = DefaultGenome(key=1)
    genome.nodes[-1] = genome.create_output_node(genome_config, -1)
    print(genome.nodes[-1])


create_simple_genome()
