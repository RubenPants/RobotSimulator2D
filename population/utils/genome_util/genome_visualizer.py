"""
visualizer.py

Create visualizations for the genomes present in the population.
"""
import os

from graphviz import Digraph

# Add graphviz to path
from population.utils.genome_util.genes import DefaultNodeGene, GruNodeGene
from population.utils.network_util.graphs import required_for_output

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def draw_net(config, genome, debug=False, filename=None, view=True):
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
    node_names[-1] = 'proximity -90°'
    node_names[-2] = 'proximity -45°'
    node_names[-3] = 'proximity 0°'
    node_names[-4] = 'proximity 45°'
    node_names[-5] = 'proximity 90°'
    node_names[-6] = 'angular left'
    node_names[-7] = 'angular right'
    node_names[-8] = 'distance'
    node_names[0] = 'left wheel'
    node_names[1] = 'right wheel'
    
    # Visualizer specific functionality
    node_colors = dict()
    dot = Digraph(format='png', engine="neato")
    dot.attr(overlap='false')
    
    # Visualize input nodes
    inputs = set()
    for index, key in enumerate(config.genome_config.input_keys):
        inputs.add(key)
        name = node_names.get(key)
        dot.node(
                name,
                style='filled',
                shape='box',
                fillcolor=node_colors.get(key, 'lightgray'),
                pos=f"{index * 3},0!"
        )
    
    # Visualize output nodes
    outputs = set()
    for index, key in enumerate(config.genome_config.output_keys):
        outputs.add(key)
        name = node_names.get(key)
        if debug:
            name += f'\nactivation={genome.nodes[key].activation}'
            name += f'\nbias={round(genome.nodes[key].bias, 2)}'
            name += f'\naggregation={genome.nodes[key].aggregation}'
        node_names.update({key: name})
        dot.node(
                name,
                style='filled',
                fillcolor=node_colors.get(key, 'lightblue'),
                pos=f"{6 + index * 9},{-5 - (len(genome.nodes) - 2) * (5 if debug else 1)}!"
        )
    
    # Visualize hidden nodes
    used_nodes, used_conn = required_for_output(
            inputs=config.genome_config.input_keys,
            outputs=config.genome_config.output_keys,
            connections=genome.connections
    )
    
    for key in used_nodes:
        if key in inputs or key in outputs: continue
        if debug:
            if type(genome.nodes[key]) == GruNodeGene:
                name = f'GRU node={key}'
                name += f'\ninputs_size={len(genome.nodes[key].input_keys)}'
                name += f'\nhidden_size={genome.nodes[key].hidden_size}'
                # name += f'\nbias_ih={np.asarray(genome.nodes[key].bias_ih.tolist()).round(3).tolist()}'
                # name += f'\nbias_hh={np.asarray(genome.nodes[key].bias_hh.tolist()).round(3).tolist()}'
                # name += f'\nweight_ih={np.asarray(genome.nodes[key].weight_ih.tolist()).round(3).tolist()}'
                # name += f'\nweight_hh={np.asarray(genome.nodes[key].weight_hh.tolist()).round(3).tolist()}'
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
                fillcolor=node_colors.get(key, 'white')
        )
    
    # Add inputs to used_nodes (i.e. all inputs will always be visualized, even if they aren't used!)
    used_nodes.update(inputs)
    
    # Visualize connections
    for cg in used_conn.values():
        sending_node, receiving_node = cg.key
        if sending_node in used_nodes and receiving_node in used_nodes:
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
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
