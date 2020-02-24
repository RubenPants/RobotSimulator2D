"""
visualizer.py

Create visualizations for the genomes present in the population.
"""
import copy
import os

from graphviz import Digraph

# Add graphviz to path
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
    # Assign names to sensors
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
            name += f'\nresponse={round(genome.nodes[key].response, 2)}'
            name += f'\naggregation={genome.nodes[key].aggregation}'
        node_names.update({key: name})
        dot.node(
                name,
                style='filled',
                fillcolor=node_colors.get(key, 'lightblue'),
                pos=f"{6 + index * 9},{-5 - (len(genome.nodes) - 2) * (5 if debug else 1)}!"
        )
    
    # Prune unused hidden nodes
    connections = set()
    for cg in genome.connections.values():
        if cg.enabled:
            connections.add(cg.key)
    
    # In the beginning, the only certainty is that the outputs are used nodes
    used_nodes = copy.copy(outputs)
    # 'pending' is used to refer to the receiving end of a connection
    pending = copy.copy(outputs)
    
    # The idea is to loop over the connections, starting from the known used_nodes going up higher in the network
    # (towards the inputs). If a node is connected to at least one the output-nodes (albeit indirectly), we know the
    # node is used by the network, and thus must be visualized
    while pending:
        new_pending = set()
        for index, connection in connections:
            if connection in pending and index not in used_nodes:
                new_pending.add(index)
                used_nodes.add(index)
        pending = new_pending
    
    # Visualize hidden nodes
    for key in used_nodes:
        if key in inputs or key in outputs:
            continue
        if debug:
            name = f'hidden node={key}'
            name += f'\nactivation={genome.nodes[key].activation}'
            name += f'\nbias={round(genome.nodes[key].bias, 2)}'
            name += f'\nresponse={round(genome.nodes[key].response, 2)}'
            name += f'\naggregation={genome.nodes[key].aggregation}'
        else:
            name = str(key)
        node_names.update({key: name})
        dot.node(name,
                 style='filled',
                 fillcolor=node_colors.get(key, 'white'))
    
    # Add inputs to used_nodes (i.e. all inputs will always be visualized, even if they aren't used!)
    used_nodes.update(inputs)
    
    # Visualize connections
    for cg in genome.connections.values():
        if cg.enabled:
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
