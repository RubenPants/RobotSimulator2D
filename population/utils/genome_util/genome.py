"""
genome.py

Handles genomes (individuals in the population). A single genome has two types of genes:
 * node gene: specifies the configuration of a single node (e.g. activation function)
 * connection gene: specifies a single connection between two neurons (e.g. weight)
"""
from __future__ import division, print_function

from random import choice, random, shuffle

from neat.six_util import iteritems, iterkeys

from configs.bot_config import BotConfig
from configs.genome_config import GenomeConfig
from environment.entities.robots import get_snapshot
from population.utils.genome_util.genes import ConnectionGene, GruNodeGene, OutputNodeGene, SimpleNodeGene
from population.utils.network_util.graphs import creates_cycle, required_for_output


class Genome(object):
    """
    A genome for generalized neural networks.

    Terminology
        pin: Point at which the network is conceptually connected to the external world; pins are either input or
            output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's output and a pin/node
            input.
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique neuron by an implicit connection with
            weight one. This connection is permanently enabled.
        2. The output pin's key is always the same as the key for its associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """
    
    def __init__(self, key, num_outputs, bot_config: BotConfig):
        # Unique identifier for a genome instance.
        self.key = key
        
        # Placeholders for the elementary parts of a genomes
        self.connections: dict = dict()  # Container for all the connections present in the genome
        self.nodes: dict = dict()  # Container for all the nodes (hidden and output)
        self.num_outputs: int = num_outputs  # Number of outputs in the genome
        self.fitness = None  # Container for the genome's fitness
        
        # Get snapshot of current robot configuration
        self.robot_snapshot = get_snapshot(cfg=bot_config)
    
    def configure_new(self, config: GenomeConfig):
        """Configure a new genome based on the given configuration."""
        # Create node genes for the output pins
        for node_key in config.keys_output: self.nodes[node_key] = self.create_output_node(config, node_key)
        
        # Add connections based on initial connectivity type
        if 'fs_neat' in config.initial_connection:
            if config.initial_connection == 'fs_neat_nohidden':
                self.connect_fs_neat_nohidden(config)
            elif config.initial_connection == 'fs_neat_hidden':
                self.connect_fs_neat_hidden(config)
            else:
                self.connect_fs_neat_nohidden(config)
        elif 'full' in config.initial_connection:
            if config.initial_connection == 'full_nodirect':
                self.connect_full_nodirect(config)
            elif config.initial_connection == 'full_direct':
                self.connect_full_direct(config)
            else:
                self.connect_full_nodirect(config)
        elif 'partial' in config.initial_connection:
            if config.initial_connection == 'partial_nodirect':
                self.connect_partial_nodirect(config)
            elif config.initial_connection == 'partial_direct':
                self.connect_partial_direct(config)
            else:
                self.connect_partial_nodirect(config)
    
    def configure_crossover(self, config: GenomeConfig, genome1, genome2):
        """Configure a new genome by crossover from two parent genomes."""
        # Rank the parents based on fitness
        assert isinstance(genome1.fitness, (int, float))  # (key, fitness)
        assert isinstance(genome2.fitness, (int, float))
        if genome1.fitness >= genome2.fitness:
            p1, p2 = genome1, genome2
        else:
            p1, p2 = genome2, genome1
        
        # Get the fitness ratio of the two parents (determines from which parent a child is most likely to inherit from)
        #  If very similar fitness values, ratio will be fixed to 0.5 (prevent division by ~0+0)
        ratio = 0.5 if abs(p1.fitness - p2.fitness) < 0.001 else p1.fitness / (p1.fitness + p2.fitness)
        
        # Inherit connection genes of the most fit genome, crossover the connection if present at both parents
        for key, cg1 in iteritems(p1.connections):
            cg2 = p2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy(cfg=config)
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cfg=config,
                                                      other=cg2,
                                                      ratio=ratio)
        
        # Inherit node genes
        for key, ng1 in iteritems(p1.nodes):
            ng2 = p2.nodes.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy(cfg=config)
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(cfg=config,
                                                other=ng2,
                                                ratio=ratio)
        
        # Make sure that all GRU-nodes are correctly configured (input_keys)
        self.update_gru_nodes(config)
    
    def mutate(self, config: GenomeConfig):
        """Mutates this genome."""
        if random() < config.node_add_prob: self.mutate_add_node(config)
        if random() < config.node_delete_prob: self.mutate_delete_node(config)
        if random() < config.conn_add_prob: self.mutate_add_connection(config)
        if random() < config.conn_delete_prob: self.mutate_delete_connection()
        
        # Mutate connection genes
        for cg in self.connections.values():
            mut_enabled = cg.mutate(config)
            if mut_enabled is not None:
                self.enable_connection(config=config, conn=cg) if mut_enabled else self.disable_connection(conn=cg)
        
        # Mutate node genes (bias etc.)
        for ng in self.nodes.values():
            ng.mutate(config)
    
    def mutate_add_node(self, config: GenomeConfig):
        """Add (or enable) a node as part of a mutation."""
        if not self.connections: return
        
        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        node_id = config.get_new_node_key(self.nodes)
        
        # Choose type of node to mutate to and add the node, must be done before adding the connection!
        r = random()
        if config.gru_enabled and r <= config.gru_node_prob:
            ng = self.create_gru_node(config, node_id)  # New nodes only have 1 ingoing connection
        else:
            ng = self.create_node(config, node_id)
        self.nodes[node_id] = ng
        
        # Disable this connection and create two new connections joining its nodes via the given node. The first
        # connection will simply forward its inputs (i.e. weight=1.0), whereas the second connection tries to mimic the
        # original (split) connection.
        self.disable_connection(conn_to_split)
        i, o = conn_to_split.key
        self.create_connection(config=config, input_key=i, output_key=node_id, weight=1.0)
        self.create_connection(config=config, input_key=node_id, output_key=o, weight=conn_to_split.weight)
    
    def mutate_delete_node(self, config: GenomeConfig):
        """Delete (disable) a node as part of a mutation."""
        # Do nothing if there are no non-output nodes.
        available_nodes = [k for k in iterkeys(self.nodes) if k not in config.keys_output]
        if not available_nodes:
            return
        
        del_key = choice(available_nodes)
        connections_to_delete = set()
        for k, v in iteritems(self.connections):
            if del_key in v.key:
                connections_to_delete.add(v.key)
        
        for key in connections_to_delete:
            self.disable_connection(self.connections[key])
            del self.connections[key]
        
        del self.nodes[del_key]
    
    def create_connection(self, config: GenomeConfig, input_key: int, output_key: int, weight: float = None):
        """Add a connection to the genome."""
        # Create the connection
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0  # output_key is not one of the inputs (sensor readings)
        assert input_key not in config.keys_output
        key = (input_key, output_key)
        connection = ConnectionGene(key, cfg=config)
        
        if weight:
            assert isinstance(weight, float)
            connection.weight = weight
        
        self.enable_connection(config=config, conn=connection)
        self.connections[key] = connection
    
    def mutate_add_connection(self, config: GenomeConfig):
        """
        Attempt to add a new connection. A connection starts in the input_node and ends in the output_node.
        The restrictions laid on the mutation are:
         - An output of the network may never be an input_node (sending-end)
         - An input of the network may never be an output_node (receiving-end)
        """
        # List all the keys that are possible output nodes (i.e. all output and hidden nodes)
        possible_outputs = list(iterkeys(self.nodes))
        out_node = choice(possible_outputs)
        
        # List all the keys that are possible input-nodes (i.e. all input and hidden nodes)
        possible_inputs = [i for i in possible_outputs + config.keys_input if i not in config.keys_output]
        in_node = choice(possible_inputs)
        
        # Don't duplicate connections
        key = (in_node, out_node)
        if key in self.connections:
            self.enable_connection(config=config, conn=self.connections[key])
            return
        
        # Avoid creating cycles.
        if creates_cycle(list(iterkeys(self.connections)), key):
            return
        
        # Create the new connection
        self.create_connection(config, in_node, out_node)
    
    def mutate_delete_connection(self):
        """Delete the connection as part of a mutation."""
        if self.connections:
            key = choice(list(self.connections.keys()))
            self.disable_connection(self.connections[key])
            del self.connections[key]
    
    def enable_connection(self, config: GenomeConfig, conn: ConnectionGene):
        """Enable the connection, and ripple this through to its potential GRU cell."""
        conn.enabled = True
        if type(self.nodes[conn.key[1]]) == GruNodeGene:
            gru: GruNodeGene = self.nodes[conn.key[1]]
            gru.add_input_key(config, k=conn.key[0])
    
    def disable_connection(self, conn: ConnectionGene):
        """Disable the connection, and ripple this through to its potential GRU cell."""
        conn.enabled = False
        if type(self.nodes[conn.key[1]]) == GruNodeGene:
            gru: GruNodeGene = self.nodes[conn.key[1]]
            gru.remove_input_key(k=conn.key[0])
    
    def size(self):
        """Returns genome 'complexity', taken to be (number of hidden nodes, number of enabled connections)"""
        inputs = {a for (a, _) in self.connections if a < 0}
        used_nodes, used_conn = required_for_output(
                inputs=inputs,
                outputs={i for i in range(self.num_outputs)},
                connections=self.connections,
        )
        return len(used_nodes) - (len(inputs) + self.num_outputs), len(used_conn)
    
    def __str__(self):
        s = f"Key: {self.key}\n" \
            f"Fitness: {self.fitness}\n" \
            f"Nodes:\n"
        for k, ng in sorted(self.nodes.items(), key=lambda x: x[0]):
            s += f"\t{k} - {repr(ng)!s}\n"
        s += "Connections:\n"
        for k, cg in sorted(self.connections.items(), key=lambda x:x[0]):
            if cg.enabled:
                s += f"\t{k} - {repr(cg)!s}\n"
        return s
    
    def __repr__(self):
        return f"DefaultGenome(key={self.key}, n_nodes={len(self.nodes)}, n_connections={len(self.connections)})"
    
    @staticmethod
    def create_node(config: GenomeConfig, node_id: int):
        node = SimpleNodeGene(node_id, cfg=config)
        return node
    
    @staticmethod
    def create_output_node(config: GenomeConfig, node_id: int):
        node = OutputNodeGene(node_id, cfg=config)
        return node
    
    @staticmethod
    def create_gru_node(config: GenomeConfig, node_id: int):
        node = GruNodeGene(node_id, cfg=config)
        return node
    
    def update_gru_nodes(self, config: GenomeConfig):
        """Update all the hidden GRU-nodes such that their input_keys are correct."""
        for (key, node) in self.nodes.items():
            if type(node) == GruNodeGene:
                # Get all the input-keys
                input_keys = set(a for (a, b) in self.connections.keys() if b == key)
                
                # Remove older inputs that aren't inputs anymore
                for k in reversed(node.input_keys):
                    if k not in input_keys: node.remove_input_key(k)
                
                # Add new inputs that were not yet inputs
                for k in input_keys:
                    if k not in node.input_keys: node.add_input_key(config, k)
                
                # Change in input_keys results in a change in weight_ih
                node.update_weight_ih()
                assert len(node.input_keys) == len(input_keys)
    
    def connect_fs_neat_nohidden(self, config: GenomeConfig):
        """
        Randomly connect one input to all output nodes
        (FS-NEAT without connections to hidden, if any).
        Originally connect_fs_neat.
        """
        input_id = choice(config.keys_input)
        for output_id in config.keys_output:
            self.create_connection(config, input_id, output_id)
    
    def connect_fs_neat_hidden(self, config: GenomeConfig):
        """
        Randomly connect one input to all hidden and output nodes
        (FS-NEAT with connections to hidden, if any).
        """
        input_id = choice(config.keys_input)
        others = [i for i in iterkeys(self.nodes) if i not in config.keys_input]
        for output_id in others:
            self.create_connection(config, input_id, output_id)
    
    def compute_full_connections(self, config: GenomeConfig, direct):
        """
        Compute connections for a fully-connected feed-forward genome--each input connected to all hidden nodes (and
        output nodes if ``direct`` is set or there are no hidden nodes), each hidden node connected to all output nodes.
        """
        hidden = [i for i in iterkeys(self.nodes) if i not in config.keys_output]
        output = [i for i in iterkeys(self.nodes) if i in config.keys_output]
        connections = []
        if hidden:
            for input_id in config.keys_input:
                for h in hidden:
                    connections.append((input_id, h))
            for h in hidden:
                for output_id in output:
                    connections.append((h, output_id))
        if direct or (not hidden):
            for input_id in config.keys_input:
                for output_id in output:
                    connections.append((input_id, output_id))
        
        return connections
    
    def connect_full_nodirect(self, config: GenomeConfig):
        """Create a fully-connected genome (except without direct input-output unless no hidden nodes)."""
        for input_id, output_id in self.compute_full_connections(config, False):
            self.create_connection(config, input_id, output_id)
    
    def connect_full_direct(self, config: GenomeConfig):
        """ Create a fully-connected genome, including direct input-output connections. """
        for input_id, output_id in self.compute_full_connections(config, True):
            self.create_connection(config, input_id, output_id)
    
    def connect_partial_nodirect(self, config: GenomeConfig):
        """Create a partially-connected genome, with (unless no hidden nodes) no direct input-output connections."""
        assert 0 <= config.conn_fraction <= 1
        all_connections = self.compute_full_connections(config, False)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.conn_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            self.create_connection(config, input_id, output_id)
    
    def connect_partial_direct(self, config: GenomeConfig):
        """Create a partially-connected genome, including (possibly) direct input-output connections.
        """
        assert 0 <= config.conn_fraction <= 1
        all_connections = self.compute_full_connections(config, True)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.conn_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            self.create_connection(config, input_id, output_id)
