"""
visualizer.py

Create visualizations for the genomes present in the population.
"""
import os
import sys

from graphviz import Digraph

from configs.genome_config import GenomeConfig
from population.utils.genome_util.genes import GruNodeGene, SimpleNodeGene
from population.utils.genome_util.genome import Genome
from population.utils.network_util.graphs import required_for_output

# Add graphviz to path if on Windows
if sys.platform == 'win32': os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def draw_net(config: GenomeConfig, genome: Genome, debug=False, filename=None, view=True):
    """
    Visualize the structure of one genome.
    
    :param config: Configuration of the network
    :param genome: Genome (network) that will be visualized
    :param debug: Add excessive information to the drawing
    :param filename: Name of the file
    :param view: Visualize when method is run
    """
    # Assign names to sensors (hard-coded since immutable)
    node_names = dict()
    node_names.update(genome.robot_snapshot)
    node_names[0] = 'left wheel'
    node_names[1] = 'right wheel'
    num_inputs = len(genome.robot_snapshot)
    
    # Visualizer specific functionality
    node_colors = dict()
    dot = Digraph(format='png', engine="fdp")
    dot.attr(overlap='false')
    
    # Get the used hidden nodes and all used connections
    used_inp_nodes, used_hid_nodes, used_outp_nodes, used_conn = required_for_output(
            inputs=set(config.keys_input),
            outputs=set(config.keys_output),
            connections=genome.connections
    )
    
    # Visualize input nodes
    inputs = set()
    active = {a for (a, b) in used_conn if a < 0}
    for index, key in enumerate(config.keys_input):
        inputs.add(key)
        name = node_names.get(key)
        color = '#e3e3e3' if key in active else '#9e9e9e'
        if debug or key in active:
            dot.node(
                    name,
                    style='filled',
                    shape='box',
                    fillcolor=node_colors.get(key, color),
                    pos=f"{index * 20},0!"
            )
    
    # Visualize output nodes
    outputs = set()
    for index, key in enumerate(config.keys_output):
        outputs.add(key)
        name = node_names[key]
        if debug:
            name += f'\nactivation={genome.nodes[key].activation}'
            name += f'\nbias={round(genome.nodes[key].bias, 2)}'
        node_names.update({key: name})
        dot.node(
                name,
                style='filled',
                shape='box',
                fillcolor=node_colors.get(key, '#bdc5ff'),
                pos=f"{(num_inputs - 5) * 10 + index * 100}, "
                    f"{200 + len(used_hid_nodes) * (50 if debug else 20)}!",
        )
    
    # Visualize hidden nodes
    for key in sorted(used_hid_nodes):
        assert (key not in inputs) and (key not in outputs)
        fillcolor = 'white' if type(genome.nodes[key]) == SimpleNodeGene else '#f5c484'  # fancy orange
        if debug:
            if type(genome.nodes[key]) == GruNodeGene:
                genome.update_gru_nodes(config)
            name = str(genome.nodes[key])
        else:
            name = str(key)
        node_names.update({key: name})
        dot.node(
                name,
                style='filled',
                shape='box',
                fillcolor=node_colors.get(key, fillcolor),
        )
    
    # Add inputs to used_nodes (i.e. all inputs will always be visualized, even if they aren't used!)
    used_nodes = (used_inp_nodes | used_hid_nodes | used_outp_nodes)
    
    # Visualize connections
    for cg in used_conn.values():
        sending_node, receiving_node = cg.key
        if sending_node in used_nodes and receiving_node in used_nodes:
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight) / config.weight_max_value * 5)
            dot.edge(
                    node_names.get(sending_node),
                    node_names.get(receiving_node),
                    label=str(round(cg.weight, 2)) if debug else None,
                    color=color,
                    penwidth=width,
            )
    
    # Render, save (and show if on Windows)
    if sys.platform == 'win32':
        dot.render(filename, view=view)
    else:
        dot.render(filename, view=False)
    
    # Remove graphviz file created during rendering
    os.remove(filename)
