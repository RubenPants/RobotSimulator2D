"""
gru.py

Test if the GruNodeGene is implemented correctly
"""
import os

import neat

from configs.config import NeatConfig
from population.population import query_net
from population.utils.genome_util.genome import DefaultGenome
from population.utils.network_util.recurrent_gru_net import make_net
from population.utils.config.population_config import PopulationConfig
from population.utils.population_util.reproduction import DefaultReproduction
from population.utils.population_util.species import DefaultSpeciesSet

cfg = NeatConfig()
cfg.num_inputs = 1
cfg.num_outputs = 1
cfg.num_hidden = 1
cfg.initial_connection = "full_nodirect"  # input->hidden and hidden->output
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
    """
    A simple genome has one input, one output, and one hidden recurrent node (GRU). All weight parameters are forced to
    be 1, where all biases are forced to be 0.
    """
    g = DefaultGenome(key=1)
    
    # Init with randomized configuration
    g.configure_new(genome_config)
    
    # Node -1: input
    # --> No change needed
    # Node 0: output
    g.nodes[0].bias = 0
    # Node 1: GRU
    GRU_KEY = 1
    g.nodes[GRU_KEY] = g.create_gru_node(genome_config, 1, [-1])
    # g.nodes[GRU_KEY].bias_ih[:] = torch.FloatTensor([0, 0, 0])
    # g.nodes[GRU_KEY].bias_hh[:] = torch.FloatTensor([0, 0, 0])
    # g.nodes[GRU_KEY].weight_ih[:] = torch.FloatTensor([[0], [0], [0]])
    # g.nodes[GRU_KEY].weight_hh[:] = torch.FloatTensor([[0], [0], [0]])
    # g.nodes[1].bias = 0
    
    # Connections
    for c in g.connections.values(): c.weight = 1
    return g


if __name__ == '__main__':
    os.chdir("..")
    genome = create_simple_genome()
    print(genome, end="\n" * 3)
    # print(genome.input_keys)
    net = make_net(genome, config, bs=1, cold_start=True)
    
    # Query the network
    print("Querying the network:")
    inp = query_net(net, [[1]])
    print(f" - iteration1: {inp}")
    inp = query_net(net, [[1]])
    print(f" - iteration2: {inp}")
    inp = query_net(net, [[1]])
    print(f" - iteration3: {inp}")
    # for _ in range(10):
    #     print(query_net(net, [[0]]))
