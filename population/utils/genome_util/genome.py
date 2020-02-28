"""
genome.py

Handles genomes (individuals in the population). A single genome has two types of genes:
 * node gene: specifies the configuration of a single node (e.g. activation function)
 * connection gene: specifies a single connection between two neurons (e.g. weight)
"""
from __future__ import division, print_function

import sys
from random import choice, random, shuffle

from neat.six_util import iteritems, iterkeys

from population.utils.config.genome_config import DefaultGenomeConfig
from population.utils.genome_util.genes import DefaultConnectionGene, DefaultNodeGene, GruNodeGene, OutputNodeGene
from population.utils.network_util.graphs import creates_cycle


class DefaultGenome(object):
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
    
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['gru_node_gene_type'] = GruNodeGene
        param_dict['output_node_gene_type'] = OutputNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)
    
    @classmethod
    def write_config(cls, f, config: DefaultGenomeConfig):
        config.save(f)
    
    def __init__(self, key):
        # Unique identifier for a genome instance.
        self.key = key
        
        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}
        
        # Fitness results.
        self.fitness = None
    
    def configure_new(self, config: DefaultGenomeConfig):
        """Configure a new genome based on the given configuration."""
        # Create node genes for the output pins.
        for node_key in config.output_keys: self.nodes[node_key] = self.create_output_node(config, node_key)
        
        # Add hidden nodes if requested.
        if config.num_hidden > 0:
            for i in range(config.num_hidden):
                node_key = config.get_new_node_key(self.nodes)
                assert node_key not in self.nodes
                r = random()
                if config.enable_gru and r <= config.gru_mutate_rate:
                    node = self.create_gru_node(config, node_key)  # New nodes only have 1 ingoing connection
                else:
                    node = self.create_node(config, node_key)
                self.nodes[node_key] = node
        
        # Add connections based on initial connectivity type.
        if 'fs_neat' in config.initial_connection:
            if config.initial_connection == 'fs_neat_nohidden':
                self.connect_fs_neat_nohidden(config)
            elif config.initial_connection == 'fs_neat_hidden':
                self.connect_fs_neat_hidden(config)
            else:
                if config.num_hidden > 0:
                    print("Warning: initial_connection = fs_neat will not connect to hidden nodes;",
                          "\tif this is desired, set initial_connection = fs_neat_nohidden;",
                          "\tif not, set initial_connection = fs_neat_hidden",
                          sep='\n', file=sys.stderr)
                self.connect_fs_neat_nohidden(config)
        elif 'full' in config.initial_connection:
            if config.initial_connection == 'full_nodirect':
                self.connect_full_nodirect(config)
            elif config.initial_connection == 'full_direct':
                self.connect_full_direct(config)
            else:
                if config.num_hidden > 0:
                    print("Warning: initial_connection = full with hidden nodes will not do direct input-output "
                          "connections;",
                          "\tif this is desired, set initial_connection = full_nodirect;",
                          "\tif not, set initial_connection = full_direct",
                          sep='\n', file=sys.stderr)
                self.connect_full_nodirect(config)
        elif 'partial' in config.initial_connection:
            if config.initial_connection == 'partial_nodirect':
                self.connect_partial_nodirect(config)
            elif config.initial_connection == 'partial_direct':
                self.connect_partial_direct(config)
            else:
                if config.num_hidden > 0:
                    print("Warning: initial_connection = partial with hidden nodes will not do direct input-output "
                          "connections;",
                          "\tif this is desired, set initial_connection = partial_nodirect {0};".format(
                                  config.connection_fraction),
                          "\tif not, set initial_connection = partial_direct {0}".format(
                                  config.connection_fraction),
                          sep='\n', file=sys.stderr)
                self.connect_partial_nodirect(config)
    
    def configure_crossover(self, genome1, genome2):
        """
        Configure a new genome by crossover from two parent genomes.
        
        TODO: Unstable if working with GRU-nodes! (no check on input-connections)
        """
        assert isinstance(genome1.fitness, (int, float))
        assert isinstance(genome2.fitness, (int, float))
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1
        
        # Inherit connection genes
        for key, cg1 in iteritems(parent1.connections):
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)
        
        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes
        
        for key, ng1 in iteritems(parent1_set):
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)
    
    def mutate(self, config: DefaultGenomeConfig):
        """ Mutates this genome. """
        if random() < config.node_add_prob: self.mutate_add_node(config)
        if random() < config.node_delete_prob: self.mutate_delete_node(config)
        if random() < config.conn_add_prob: self.mutate_add_connection(config)
        if random() < config.conn_delete_prob: self.mutate_delete_connection()
        
        # Mutate connection genes.
        for cg in self.connections.values():
            mut_enabled = cg.mutate(config)
            if mut_enabled is not None:
                self.enable_connection(config=config, conn=cg) if mut_enabled else self.disable_connection(conn=cg)
        
        # Mutate node genes (bias etc.).
        for ng in self.nodes.values():
            ng.mutate(config)
    
    def mutate_add_node(self, config: DefaultGenomeConfig):
        if not self.connections:
            if config.check_structural_mutation_surer(): self.mutate_add_connection(config)
            return
        
        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        node_id = config.get_new_node_key(self.nodes)
        
        # Choose type of node to mutate to and add the node, must be done before adding the connection!
        r = random()
        if config.enable_gru and r <= config.gru_mutate_rate:
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
    
    def mutate_delete_node(self, config: DefaultGenomeConfig):
        # Do nothing if there are no non-output nodes.
        available_nodes = [k for k in iterkeys(self.nodes) if k not in config.output_keys]
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
    
    def create_connection(self, config: DefaultGenomeConfig, input_key: int, output_key: int, weight: float = None):
        """Add a connection to the genome."""
        # Create the connection
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0  # output_key is not one of the inputs (sensor readings)
        assert input_key not in config.output_keys
        key = (input_key, output_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config)
        
        if weight:
            assert isinstance(weight, float)
            connection.weight = weight
        
        self.enable_connection(config=config, conn=connection)
        self.connections[key] = connection
    
    def mutate_add_connection(self, config: DefaultGenomeConfig):
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
        possible_inputs = [i for i in possible_outputs + config.input_keys if i not in config.output_keys]
        in_node = choice(possible_inputs)
        
        # Don't duplicate connections.
        key = (in_node, out_node)
        if key in self.connections:
            if config.check_structural_mutation_surer():
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
    
    def enable_connection(self, config: DefaultGenomeConfig, conn: DefaultConnectionGene):
        """Enable the connection, and ripple this through to its potential GRU cell."""
        conn.enabled = True
        if type(self.nodes[conn.key[1]]) == GruNodeGene:
            gru: GruNodeGene = self.nodes[conn.key[1]]
            gru.append_key(config, k=conn.key[0])
    
    def disable_connection(self, conn: DefaultConnectionGene):
        """Disable the connection, and ripple this through to its potential GRU cell."""
        conn.enabled = False
        if type(self.nodes[conn.key[1]]) == GruNodeGene:
            gru: GruNodeGene = self.nodes[conn.key[1]]
            gru.delete_key(k=conn.key[0])
    
    def distance(self, other, config: DefaultGenomeConfig):
        """
        Returns the genetic distance between this genome and the other. This distance value is used to compute genome
        compatibility for speciation.
        """
        # Compute node gene distance component.
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in iterkeys(other.nodes):
                if k2 not in self.nodes:
                    disjoint_nodes += 1
            
            for k1, n1 in iteritems(self.nodes):
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)
            
            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance +
                             (config.compatibility_disjoint_coefficient *
                              disjoint_nodes)) / max_nodes
        
        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in iterkeys(other.connections):
                if k2 not in self.connections:
                    disjoint_connections += 1
            
            for k1, c1 in iteritems(self.connections):
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)
            
            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_connections)) / max_conn
        
        distance = node_distance + connection_distance
        return distance
    
    def size(self):
        """Returns genome 'complexity', taken to be (number of hidden nodes, number of enabled connections)"""
        nodes = 0
        ingoing = [c[0] for c in self.connections.keys()]
        outgoing = [c[1] for c in self.connections.keys()]
        for node_key in self.nodes:
            if (node_key not in [0, 1]) and (node_key in ingoing) and (node_key in outgoing): nodes += 1
        return nodes, sum([1 for cg in self.connections.values() if cg.enabled])
    
    def __str__(self):
        s = f"Key: {self.key}\nFitness: {self.fitness}\nNodes:"
        for k, ng in iteritems(self.nodes): s += f"\n\t{k} {ng!s}"
        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections: s += "\n\t" + str(c)
        return s
    
    @staticmethod
    def create_node(config: DefaultGenomeConfig, node_id: int):
        node = config.node_gene_type(node_id)
        node.init_attributes(config)
        return node
    
    @staticmethod
    def create_output_node(config: DefaultGenomeConfig, node_id: int):
        node = config.output_node_gene_type(node_id)
        node.init_attributes(config)
        return node
    
    @staticmethod
    def create_gru_node(config: DefaultGenomeConfig, node_id: int, input_size=1):
        node = config.gru_node_gene_type(node_id)
        node.init_attributes(config)  # TODO: Update such that input_size is given!
        return node
    
    def connect_fs_neat_nohidden(self, config: DefaultGenomeConfig):
        """
        Randomly connect one input to all output nodes
        (FS-NEAT without connections to hidden, if any).
        Originally connect_fs_neat.
        """
        input_id = choice(config.input_keys)
        for output_id in config.output_keys:
            self.create_connection(config, input_id, output_id)
    
    def connect_fs_neat_hidden(self, config: DefaultGenomeConfig):
        """
        Randomly connect one input to all hidden and output nodes
        (FS-NEAT with connections to hidden, if any).
        """
        input_id = choice(config.input_keys)
        others = [i for i in iterkeys(self.nodes) if i not in config.input_keys]
        for output_id in others:
            self.create_connection(config, input_id, output_id)
    
    def compute_full_connections(self, config: DefaultGenomeConfig, direct):
        """
        Compute connections for a fully-connected feed-forward genome--each input connected to all hidden nodes (and
        output nodes if ``direct`` is set or there are no hidden nodes), each hidden node connected to all output nodes.
        """
        hidden = [i for i in iterkeys(self.nodes) if i not in config.output_keys]
        output = [i for i in iterkeys(self.nodes) if i in config.output_keys]
        connections = []
        if hidden:
            for input_id in config.input_keys:
                for h in hidden:
                    connections.append((input_id, h))
            for h in hidden:
                for output_id in output:
                    connections.append((h, output_id))
        if direct or (not hidden):
            for input_id in config.input_keys:
                for output_id in output:
                    connections.append((input_id, output_id))
        
        return connections
    
    def connect_full_nodirect(self, config: DefaultGenomeConfig):
        """Create a fully-connected genome (except without direct input-output unless no hidden nodes)."""
        for input_id, output_id in self.compute_full_connections(config, False):
            self.create_connection(config, input_id, output_id)
    
    def connect_full_direct(self, config: DefaultGenomeConfig):
        """ Create a fully-connected genome, including direct input-output connections. """
        for input_id, output_id in self.compute_full_connections(config, True):
            self.create_connection(config, input_id, output_id)
    
    def connect_partial_nodirect(self, config: DefaultGenomeConfig):
        """Create a partially-connected genome, with (unless no hidden nodes) no direct input-output connections."""
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, False)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            self.create_connection(config, input_id, output_id)
    
    def connect_partial_direct(self, config: DefaultGenomeConfig):
        """Create a partially-connected genome, including (possibly) direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, True)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            self.create_connection(config, input_id, output_id)
