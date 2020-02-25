"""
gru.py

Test if the GruNodeGene is implemented correctly
"""
import neat
import torch

from configs.config import NeatConfig
from main import make_net, query_net
from population.utils.genome import DefaultGenome
from population.utils.population_config import PopulationConfig
from population.utils.reproduction import DefaultReproduction
from population.utils.species import DefaultSpeciesSet

cfg = NeatConfig()
cfg.num_inputs = 1
cfg.num_outputs = 1
cfg.num_hidden = 1
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
    g = DefaultGenome(key=1)
    
    # Init with randomized configuration
    g.configure_new(genome_config)
    
    # Node -1: input
    # --> No change needed
    # Node 0: output
    g.nodes[0].bias = 0
    # Node 1: GRU
    g.nodes[1] = g.create_gru_node(genome_config, 1)
    g.nodes[1].bias_ih[:] = torch.FloatTensor([0, 0, 0])
    g.nodes[1].bias_hh[:] = torch.FloatTensor([0, 0, 0])
    g.nodes[1].weight_ih[:] = torch.FloatTensor([[0], [0], [0]])
    g.nodes[1].weight_hh[:] = torch.FloatTensor([[0], [0], [0]])
    
    # Connections
    for c in g.connections.values(): c.weight = 1
    return g


genome = create_simple_genome()
# print(genome.input_keys)
net = make_net(genome, config, 1)
print(net)
