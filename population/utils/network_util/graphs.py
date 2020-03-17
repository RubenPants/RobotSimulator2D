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
    Determine which nodes and connections are needed to compute the final output. It is considered that only paths
    starting at the inputs and ending at the outputs are relevant. This decision is made since a node bias can
    substitute for a 'floating' node (i.e. node with no input and constant output).
    
    :note: It is assumed that the input identifier set and the node identifier set are disjoint. By convention, the
           output node ids are always the same as the output index.
    
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    :return: Set of used nodes, Remaining connections
    """
    # Get all the enabled connections and the nodes used in those
    used_conn = {k: c for k, c in connections.items() if c.enabled}
    used_nodes = set(a for (a, _) in used_conn.keys())
    used_nodes.update({b for (_, b) in used_conn.keys()})
    used_nodes.update({n for n in inputs + outputs})
    
    # Initialize with dummy to get the 'while' going
    removed_nodes = [True]
    
    # While new nodes get removed, keep pruning
    while removed_nodes:
        removed_nodes = []
        
        # Search for nodes to prune
        for n in used_nodes:
            # Inputs and outputs cannot be pruned
            if n in inputs + outputs: continue
            
            # Node must be at least once both at the sender and the receiving side of a connection
            if not (n in {a for (a, _) in used_conn.keys()} and n in {b for (_, b) in used_conn.keys()}):
                removed_nodes.append(n)
        
        # Delete the removed_nodes from the used_nodes set, remove their corresponding connections as well
        for n in removed_nodes:
            # Remove the dangling node
            used_nodes.remove(n)
            
            # Connection must span between two used nodes
            used_conn = {(a, b): c for (a, b), c in used_conn.items() if (a in used_nodes and b in used_nodes)}
    
    # Return the set of used nodes, as well as all the remaining (used) connections
    return used_nodes, used_conn
