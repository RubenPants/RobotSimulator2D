"""
visualizer.py

Create visualizations for the genomes present in the population.
"""
import os
import numpy as np

from graphviz import Digraph

from population.utils.genome_util.genes import DefaultNodeGene, GruNodeGene
from population.utils.genome_util.genome import DefaultGenome
from population.utils.network_util.graphs import required_for_output

# Add graphviz to path
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def draw_net(config, genome: DefaultGenome, debug=False, filename=None, view=True):
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
    used_nodes, used_conn = required_for_output(
            inputs=set(config.genome_config.input_keys),
            outputs=set(config.genome_config.output_keys),
            connections=genome.connections
    )
    
    # Visualize input nodes
    inputs = set()
    active = {a for (a, b) in used_conn if a < 0}
    for index, key in enumerate(config.genome_config.input_keys):
        inputs.add(key)
        name = node_names.get(key)
        color = '#e3e3e3' if key in active else '#9e9e9e'
        dot.node(
                name,
                style='filled',
                shape='box',
                fillcolor=node_colors.get(key, color),
                pos=f"{index * 20},0!"
        )
    
    # Visualize output nodes
    outputs = set()
    for index, key in enumerate(config.genome_config.output_keys):
        outputs.add(key)
        name = node_names[key]
        if debug:
            name += f'\nactivation={genome.nodes[key].activation}'
            name += f'\nbias={round(genome.nodes[key].bias, 2)}'
            name += f'\naggregation={genome.nodes[key].aggregation}'
        node_names.update({key: name})
        dot.node(
                name,
                style='filled',
                shape='box',
                fillcolor=node_colors.get(key, '#bdc5ff'),
                pos=f"{(num_inputs - 5) * 10 + index * 100}, "
                    f"{200 + (len(used_nodes) - len(node_names)) * (50 if debug else 20)}!",
        )
    
    # Visualize hidden nodes
    for key in sorted(used_nodes):
        if key in inputs or key in outputs: continue
        fillcolor = 'white'
        if debug:
            if type(genome.nodes[key]) == GruNodeGene:
                name = f'GRU node={key}'
                name += f'\ninputs_size={len(genome.nodes[key].input_keys)}'
                name += f'\nbias_ih={np.asarray(genome.nodes[key].gru_bias_ih.tolist()).round(3).tolist()}'
                name += f'\nbias_hh={np.asarray(genome.nodes[key].gru_bias_hh.tolist()).round(3).tolist()}'
                name += f'\nweight_ih={np.asarray(genome.nodes[key].gru_weight_ih.tolist()).round(3).tolist()}'
                name += f'\nweight_hh={np.asarray(genome.nodes[key].gru_weight_hh.tolist()).round(3).tolist()}'
                fillcolor = '#f5c484'  # Fancy orange
            elif type(genome.nodes[key]) == DefaultNodeGene:
                name = f'simple node={key}'
                name += f'\nactivation={genome.nodes[key].activation}'
                name += f'\nbias={round(genome.nodes[key].bias, 2)}'
                name += f'\naggregation={genome.nodes[key].aggregation}'
            else:
                raise Exception(f"Invalid hidden node (key={key}) of genome: \n{genome}")
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
    used_nodes.update(inputs)
    
    # Visualize connections
    for cg in used_conn.values():
        sending_node, receiving_node = cg.key
        if sending_node in used_nodes and receiving_node in used_nodes:
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight * 5))
            dot.edge(
                    node_names.get(sending_node),
                    node_names.get(receiving_node),
                    label=str(round(cg.weight, 2)) if debug else None,
                    color=color,
                    penwidth=width,
            )
    
    # Render and save
    dot.render(filename, view=view)
    
    # Remove graphviz file created during rendering
    os.remove(filename)
