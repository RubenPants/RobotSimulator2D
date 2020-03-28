"""
feed_forward_net.py

Create a simple feedforward network. The network will propagate one iteration at a time, doing the following:
 1) Update the hidden nodes their state taking the input and other hidden nodes into account
 2) Execute the GRUs such that their current state is updated (input=current_state)
 2) Update the output nodes their state taking the input and hidden nodes into account

Copyright (c) 2018 Uber Technologies, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import numpy as np
import torch

from config import Config
from configs.genome_config import GenomeConfig
from environment.entities.robots import get_active_sensors
from population.utils.genome_util.genes import GruNodeGene
from population.utils.genome_util.genome import Genome
from population.utils.network_util.activations import relu_activation, tanh_activation
from population.utils.network_util.graphs import required_for_output
from population.utils.network_util.shared import dense_from_coo


class FeedForwardNet:
    def __init__(self,
                 input_idx, hidden_idx, gru_idx, output_idx,
                 in2hid, in2out,
                 hid2hid, hid2out,
                 hidden_biases, output_biases,
                 grus, gru_map,
                 game_config: Config,
                 batch_size=1,
                 hidden_activation=relu_activation, output_activation=tanh_activation,  # TODO: Make configurable?
                 initial_read: list = None,
                 dtype=torch.float64,
                 ):
        """
        Create a simple feedforward network used as the control-mechanism for the drones.
        
        :param input_idx: Input indices (sensors)
        :param hidden_idx: Hidden simple-node indices (DefaultGeneNode) in the network
        :param gru_idx: Hidden GRU-node indices (DefaultGeneNode) in the network
        :param output_idx: Output indices (the two differential wheels)
        :param in2hid: Connections connecting the input nodes to the hidden nodes
        :param in2out: Connections directly connecting from the inputs to the outputs
        :param hid2hid: Connections between the hidden nodes
        :param hid2out: Connections from hidden nodes towards the outputs
        :param grus: List of GRUCell objects (length equals len(gru_idx))
        :param gru_map: Boolean matrix mapping raw inputs to inputs used by GRUCell for a single batch
        :param game_config: GameConfig object
        :param batch_size: Needed to setup network-dimensions
        :param hidden_activation: The default hidden-node activation function (squishing)
        :param output_activation: The default output-node activation function (squishing)
        :param initial_read: Initial sensory-input used to warm-up the network (no warm-up if None)
        :param dtype: Value-type used in the tensors
        """
        # Storing the input arguments (needed later on)
        self.hidden_act_f = hidden_activation
        self.output_act_f = output_activation
        self.dtype = dtype
        self.input_idx = input_idx
        self.hidden_idx = hidden_idx
        self.gru_idx = gru_idx
        self.output_idx = output_idx
        self.n_inputs = len(input_idx)
        self.n_hidden = len(hidden_idx)
        self.n_gru = len(gru_idx)
        self.n_outputs = len(output_idx)
        self.bs = batch_size
        self.game_config = game_config
        
        # Setup the gru_map
        self.gru_map = []
        for i, m in enumerate(gru_map):
            self.gru_map.append(np.tile(gru_map[i], (batch_size, 1)))
        self.gru_map = torch.tensor(self.gru_map, dtype=bool)
        
        # Placeholders, initialized during reset
        self.gru_cache = None  # Inputs for the GRUs before iterating through the hidden nodes
        self.gru_state = None  # State of the GRUs
        self.hidden_act = None  # Activations of the hidden nodes
        self.output_act = None  # Activations of the output nodes
        
        # Do not create the hidden-related matrices if hidden-nodes do not exist
        #  If they do not exist, a single matrix directly mapping inputs to outputs is only used
        if self.n_hidden > 0:
            self.in2hid = dense_from_coo((self.n_hidden, self.n_inputs), in2hid, dtype=dtype)
            self.hid2hid = dense_from_coo((self.n_hidden, self.n_hidden), hid2hid, dtype=dtype)
            self.hid2out = dense_from_coo((self.n_outputs, self.n_hidden), hid2out, dtype=dtype)
            self.grus = grus
        self.in2out = dense_from_coo((self.n_outputs, self.n_inputs), in2out, dtype=dtype)
        
        # Fill in the biases
        if self.n_hidden > 0: self.hidden_biases = torch.tensor(hidden_biases, dtype=dtype)
        self.output_biases = torch.tensor(output_biases, dtype=dtype)
        
        # Put network to initial (default) state
        self.reset(initial_read=initial_read)
    
    def reset(self, initial_read: list = None):
        """
        Set the network back to initial state.
        
        :param initial_read: Initial sensory-input used to warm-up the network (no warm-up if None)
        """
        # Reset the network back to zero inputs
        self.gru_cache = torch.zeros(self.bs, self.n_gru, self.n_inputs + self.n_hidden)
        self.gru_state = torch.zeros(self.bs, self.n_gru, 1)  # GRU outputs are single float
        self.hidden_act = torch.zeros(self.bs, self.n_hidden, dtype=self.dtype) if self.n_hidden > 0 else None
        self.output_act = torch.zeros(self.bs, self.n_outputs, dtype=self.dtype)
        
        # Initialize the network on maximum sensory inputs
        if initial_read:
            for _ in range(self.n_hidden):  # Worst case: path-length equals number of hidden nodes
                # Code below is straight up stolen from 'activate(self, inputs)'
                with torch.no_grad():
                    inputs = torch.tensor([initial_read] * self.bs, dtype=self.dtype)
                    output_inputs = self.in2out.mm(inputs.t()).t()
                    self.hidden_act = self.hidden_act_f(self.in2hid.mm(inputs.t()).t() +
                                                        self.hid2hid.mm(self.hidden_act.t()).t() +
                                                        self.hidden_biases)
                    output_inputs += self.hid2out.mm(self.hidden_act.t()).t()
                    self.output_act = self.output_act_f(output_inputs + self.output_biases)
    
    def activate(self, inputs):
        """
        Activate the network. This is used during the call of "query-net". It will feed the inputs into the network and
        return the resulting outputs.
        
        :param inputs: (batch_size, n_inputs)
        :return: The output-values (i.e. floats for the differential wheels) of shape (batch_size, n_outputs)
        """
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=self.dtype)  # Read in the inputs as a tensor
            
            # Denote the impact the inputs have directly on the outputs
            output_inputs = self.in2out.mm(inputs.t()).t()
            
            # Denote the impact hidden nodes have on the outputs, if there are hidden nodes
            if self.n_hidden > 0:
                # Nice to know:
                #  - tensor.t() will transpose the tensor
                #  - tensor.mm(tensor2) will perform a matrix multiplication between tensor and tensor2
                
                # The activation is defined by:
                #  - the inputs mapping to the hidden nodes
                #  - the hidden nodes mapping to themselves
                #  - the hidden nodes' biases
                
                # 0) Store the GRU inputs
                for i, gru_idx in enumerate(self.gru_idx):
                    self.gru_cache[:, i] = torch.cat((self.in2hid[gru_idx] * inputs,
                                                      self.hid2hid[gru_idx] * self.hidden_act), dim=1)
                
                # 1) Propagate the hidden nodes
                self.hidden_act = self.hidden_act_f(self.in2hid.mm(inputs.t()).t() +
                                                    self.hid2hid.mm(self.hidden_act.t()).t() +
                                                    self.hidden_biases)
                
                # 2) Execute the GRU nodes if they exists (updating current hidden state)
                for i, gru_idx in enumerate(self.gru_idx):
                    self.gru_state[:, i] = self.grus[i](
                            self.gru_cache[:, i][self.gru_map[i]].reshape(self.bs, self.grus[i].input_size),
                            self.gru_state[:, i],
                    )
                    self.hidden_act[:, gru_idx] = self.gru_state[:, i, 0]
                
                # 3) Propagate hidden-values to the outputs
                output_inputs += self.hid2out.mm(self.hidden_act.t()).t()
            
            # Define the values of the outputs, which is the sum of their received inputs and their corresponding bias
            self.output_act = self.output_act_f(output_inputs + self.output_biases)
        return self.output_act
    
    @staticmethod
    def create(genome,
               genome_config: GenomeConfig,
               game_config: Config,
               batch_size: int = 1,
               initial_read: list = None,
               logger=None,
               ):
        """
        This class will unravel the genome and create a feed-forward network based on it. In other words, it will create
        the phenotype (network) suiting the given genome.
        
        :param genome: The genome for which a network must be created
        :param genome_config: GenomeConfig object
        :param game_config: Config object
        :param batch_size: Batch-size needed to setup network dimension
        :param initial_read: Initial sensory-input used to warm-up the network (no warm-up if None)
        :param logger: A population's logger
        """
        # Collect the nodes whose state is required to compute the final network output(s), this excludes the inputs
        used_input = {a for a in get_active_sensors(connections=genome.get_used_connections(),
                                                    total_input_size=genome_config.num_inputs)}
        used_input_keys = {a - genome_config.num_inputs for a in used_input}
        used_nodes, used_conn = required_for_output(
                inputs=used_input_keys,
                outputs=set(genome_config.keys_output),
                connections=genome.connections
        )
        if initial_read: assert len(used_input) == len(initial_read)
        
        # Get a list of all the (used) input, (used) hidden, and output keys
        input_keys = sorted(used_input_keys)
        hidden_keys = [k for k in genome.nodes.keys() if (k not in genome_config.keys_output and k in used_nodes)]
        gru_keys = [k for k in hidden_keys if type(genome.nodes[k]) == GruNodeGene]
        output_keys = list(genome_config.keys_output)
        
        # Define the biases, note that inputs do not have a bias (since they aren't actually nodes!)
        hidden_biases = [genome.nodes[k].bias for k in hidden_keys]
        output_biases = [genome.nodes[k].bias for k in output_keys]
        
        # Create a mapping of a node's key to their index in their corresponding list
        input_key_to_idx = {k: i for i, k in enumerate(input_keys)}
        hidden_key_to_idx = {k: i for i, k in enumerate(hidden_keys)}
        output_key_to_idx = {k: i for i, k in enumerate(output_keys)}
        
        def key_to_idx(k):
            """Convert key to their corresponding index."""
            return input_key_to_idx[k] if k in input_keys \
                else output_key_to_idx[k] if k in output_keys \
                else hidden_key_to_idx[k]
        
        # Position-encode (index) the keys
        input_idx = [key_to_idx(k) for k in input_keys]
        hidden_idx = [key_to_idx(k) for k in hidden_keys]
        gru_idx = [key_to_idx(k) for k in gru_keys]
        output_idx = [key_to_idx(k) for k in output_keys]
        
        # Only feed-forward connections considered, these lists contain the connections and their weights respectively
        #  Note that the connections are index-based and not key-based!
        in2hid = ([], [])
        hid2hid = ([], [])
        in2out = ([], [])
        hid2out = ([], [])
        
        # Convert the key-based connections to index-based connections one by one, also save their weights
        #  At this point, it is already known that all connections are used connections
        for conn in used_conn.values():
            # Convert to index-based
            i_key, o_key = conn.key
            i_idx = key_to_idx(i_key)
            o_idx = key_to_idx(o_key)
            
            # Store
            if i_key in input_keys and o_key in hidden_keys:
                idxs, vals = in2hid
            elif i_key in hidden_keys and o_key in hidden_keys:
                idxs, vals = hid2hid
            elif i_key in input_keys and o_key in output_keys:
                idxs, vals = in2out
            elif i_key in hidden_keys and o_key in output_keys:
                idxs, vals = hid2out
            else:
                msg = f"{genome}" \
                      f"\ni_key: {i_key}, o_key: {o_key}" \
                      f"\ni_key in input_keys: {i_key in input_keys}" \
                      f"\ni_key in hidden_keys: {i_key in hidden_keys}" \
                      f"\ni_key in output_keys: {i_key in output_keys}" \
                      f"\no_key in input_keys: {o_key in input_keys}" \
                      f"\no_key in hidden_keys: {o_key in hidden_keys}" \
                      f"\no_key in output_keys: {o_key in output_keys}"
                logger(msg) if logger else print(msg)
                raise ValueError(f'Invalid connection from key {i_key} to key {o_key}')
            
            # Append to the lists of the right tuple
            idxs.append((o_idx, i_idx))  # Connection: to, from
            vals.append(conn.weight)  # Connection: weight
        
        # Create the gru-cells and put them in a list
        grus = []
        gru_map = []
        for gru_key in gru_keys:
            # Query the node that contains the GRUCell's weights
            node: GruNodeGene = genome.nodes[gru_key]
            
            # Create a map of all inputs/hidden nodes to the ones used by the GRUCell (as inputs)
            mapping = np.asarray([], dtype=bool)
            for k in input_keys: mapping = np.append(mapping, True if k in node.input_keys else False)
            for k in hidden_keys: mapping = np.append(mapping, True if k in node.input_keys else False)
            weight_map = np.asarray([k in input_keys + hidden_keys for k in node.input_keys])
            
            # Add the GRUCell and its corresponding mapping to the list of used GRUCells
            grus.append(node.get_gru(mapping=weight_map))
            assert len(mapping[mapping]) == grus[-1].input_size
            gru_map.append(mapping)
        
        return FeedForwardNet(
                input_idx=input_idx, hidden_idx=hidden_idx, gru_idx=gru_idx, output_idx=output_idx,
                in2hid=in2hid, in2out=in2out,
                hid2hid=hid2hid, hid2out=hid2out,
                hidden_biases=hidden_biases, output_biases=output_biases,
                grus=grus, gru_map=gru_map,
                game_config=game_config,
                batch_size=batch_size,
                initial_read=initial_read,
        )


def make_net(genome: Genome, genome_config: GenomeConfig, game_config: Config, bs=1, initial_read: list = None):
    """
    Create the "brains" of the candidate, based on its genetic wiring.

    :param genome: Genome specifies the brains internals
    :param genome_config: GenomeConfig object
    :param game_config: Current Config object
    :param bs: Batch size, which represents amount of games trained in parallel
    :param initial_read: Initial sensory-input used to warm-up the network (no warm-up if None)
    """
    return FeedForwardNet.create(
            genome,
            genome_config=genome_config,
            game_config=game_config,
            batch_size=bs,
            initial_read=initial_read,
    )
