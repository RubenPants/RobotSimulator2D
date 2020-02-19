"""
visualizer.py

Create visualizations for the genomes present in the population.
"""
import copy
import os

from graphviz import Digraph

# Add graphviz to path
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def draw_net(config, genome, view=True, filename=None):
    """
    Visualize the structure of one genome.
    
    :param config: Configuration of the network
    :param genome: Genome (network) that will be visualized
    :param view: Visualize when method is run
    :param filename: Name of the file
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
    for a, k in enumerate(config.genome_config.input_keys):
        inputs.add(k)
        name = node_names.get(k, str(k))
        dot.node(name,
                 style='filled',
                 shape='box',
                 fillcolor=node_colors.get(k, 'lightgray'),
                 pos="{},0!".format(a * 1.5))
    
    # Visualize output nodes
    outputs = set()
    for a, k in enumerate(config.genome_config.output_keys):
        outputs.add(k)
        name = node_names.get(k, str(k))
        dot.node(name,
                 style='filled',
                 fillcolor=node_colors.get(k, 'lightblue'),
                 pos="{},-5!".format(3.5 + a * 3))
    
    # Prune unused hidden nodes
    connections = set()
    for cg in genome.connections.values():
        if cg.enabled:
            connections.add(cg.key)
    
    used_nodes = copy.copy(outputs)
    pending = copy.copy(outputs)
    while pending:
        new_pending = set()
        for a, b in connections:
            if b in pending and a not in used_nodes:
                new_pending.add(a)
                used_nodes.add(a)
        pending = new_pending
    
    # Visualize hidden nodes
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue
        dot.node(str(n),
                 style='filled',
                 fillcolor=node_colors.get(n, 'white'))
    
    # Add inputs to used_nodes
    used_nodes.update(inputs)
    
    # Visualize connections
    for cg in genome.connections.values():
        if cg.enabled:
            a, b = cg.key
            if a in used_nodes and b in used_nodes:
                color = 'green' if cg.weight > 0 else 'red'
                width = str(0.1 + abs(cg.weight / 5.0))
                dot.edge(node_names.get(a, str(a)),
                         node_names.get(b, str(b)),
                         color=color,
                         penwidth=width)
    
    # Render and save
    dot.render(filename, view=view)
    
    # Remove graphviz file created during rendering
    os.remove(filename)
