"""
graphs.py

Directed graph algorithm implementations.
"""


def creates_cycle(connections, test):
    """
    Returns true if the addition of the 'test' connection would create a cycle, assuming that no cycle already exists in
    the graph represented by 'connections'.
    
    :param connections: List of connections, with each connection a tuple of form (sender, receiver)
    :param test: Newly added connection, represented by a tuple of form (sender, receiver)
    """
    i, o = test
    if i == o: return True
    
    visited = {o}
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == i:
                    return True
                
                visited.add(b)
                num_added += 1
        
        if num_added == 0:
            return False


def required_for_output(inputs, outputs, connections):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    
    :note: It is assumed that the input identifier set and the node identifier set are disjoint. By convention, the
           output node ids are always the same as the output index.
    :note: Only paths starting at the inputs and ending at the outputs are allowed (i.e. no floating nodes).
    
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    :return: Set of identifiers of required nodes
    """
    required = set(outputs)
    s = set(outputs)
    while 1:
        # Find nodes not in S whose output is consumed by a node in s.
        t = set(a for ((a, b), c) in connections.items() if c.enabled and b in s and a not in s)
        if not t: break
        
        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes: break
        
        required = required.union(layer_nodes)
        s = s.union(t)
    
    # Prune floating nodes (i.e. hidden nodes without inputs)
    floating = set()
    i_keys, o_keys = zip(*[k for (k, c) in connections.items() if c.enabled])
    for r in required:
        if r in inputs + outputs: continue
        if not (r in i_keys and r in o_keys): floating.add(r)
    for f in floating: required.remove(f)
    return required


def feed_forward_layers(inputs, outputs, connections):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    
    :note: The returned layers do not contain nodes whose output is ultimately never used to compute the final network
           output.
    
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    :return: List of layers, with each layer consisting of a set of node identifiers.
    """
    required = required_for_output(inputs, outputs, connections)
    
    layers = []
    s = set(inputs)
    while 1:
        # Find candidate nodes c for the next layer, these nodes should connect a node in s to a node not in s
        c = set(b for (a, b) in connections if a in s and b not in s)
        # Keep only the used nodes whose entire input set is contained in s
        t = set()
        for n in c:
            if n in required and all(a in s for (a, b) in connections if b == n): t.add(n)
        if not t: break
        
        layers.append(t)
        s = s.union(t)
    return layers
