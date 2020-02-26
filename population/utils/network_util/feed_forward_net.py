# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import numpy as np
import torch

from environment.entities.game import initial_sensor_readings
from population.utils.network_util.activations import tanh_activation
from population.utils.network_util.graphs import required_for_output


# noinspection PyArgumentList
def dense_from_coo(shape, conns, dtype=torch.float64):
    """
    Create a dense matrix based on the coordinates it will represent.
    
    :param shape: Tuple (output_size, input_size) ~ (nr_rows, nr_cols)
    :param conns: The connections that are being represented by the matrix, these connections are index-based
    :param dtype: Tensor type
    :return: PyTorch tensor
    """
    # Initialize an empty matrix of correct shape
    mat = torch.zeros(shape, dtype=dtype)
    # Split the connections-tuple in its corresponding indices- and weight-lists
    idxs, weights = conns
    # If no indices (i.e. no connections), return the empty matrix
    if len(idxs) == 0: return mat
    # Split the idxs (e.g. [(A, B)]) to rows ([A]) and cols ([B])
    rows, cols = np.array(idxs).transpose()
    # Put the weights on the correct spots in the empty tensor
    mat[torch.LongTensor(rows), torch.LongTensor(cols)] = torch.tensor(weights, dtype=dtype)
    return mat


class FeedForwardNet:
    def __init__(self,
                 n_inputs, n_hidden, n_outputs,
                 input_to_hidden, input_to_output, hidden_to_hidden, hidden_to_output,
                 hidden_biases, output_biases,
                 batch_size=1,
                 activation=tanh_activation,
                 dtype=torch.float64,
                 ):
        """
        Create a simple feedforward network used as the control-mechanism for the drones.
        
        :param n_inputs: Number of inputs (sensors)
        :param n_hidden: Number of hidden simple-nodes (DefaultGeneNode) in the network
        :param n_outputs: Number of outputs (the two differential wheels)
        :param input_to_hidden: Connections connecting the input nodes to the hidden nodes
        :param input_to_output: Connections directly connecting from the inputs to the outputs
        :param hidden_to_hidden: Connections between the hidden nodes
        :param hidden_to_output: Connections from hidden nodes towards the outputs
        :param batch_size: Needed to setup network-dimensions
        :param activation: The default node-activation function (squishing)
        :param dtype: Value-type used in the tensors
        """
        # Storing the input arguments (needed later on)
        self.activation = activation
        self.dtype = dtype
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        
        # Placeholders, initialized during reset
        self.activations = None
        self.outputs = None
        
        # Do not create the hidden-related matrices if hidden-nodes do not exist
        #  If they do not exist, a single matrix directly mapping inputs to outputs is only used
        if n_hidden > 0:
            self.input_to_hidden = dense_from_coo((n_hidden, n_inputs), input_to_hidden, dtype=dtype)
            self.hidden_to_hidden = dense_from_coo((n_hidden, n_hidden), hidden_to_hidden, dtype=dtype)
            self.hidden_to_output = dense_from_coo((n_outputs, n_hidden), hidden_to_output, dtype=dtype)
        self.input_to_output = dense_from_coo((n_outputs, n_inputs), input_to_output, dtype=dtype)
        
        # Fill in the biases
        if n_hidden > 0: self.hidden_biases = torch.tensor(hidden_biases, dtype=dtype)
        self.output_biases = torch.tensor(output_biases, dtype=dtype)
        
        # Put network to initial (default) state
        self.initial_readings = initial_sensor_readings()
        self.reset(batch_size)
    
    def reset(self, batch_size=1):
        """Set the network back to initial state."""
        # Reset the network back to zero inputs
        self.activations = torch.zeros(batch_size, self.n_hidden, dtype=self.dtype) if self.n_hidden > 0 else None
        self.outputs = torch.zeros(batch_size, self.n_outputs, dtype=self.dtype)
        
        # Initialize the network on maximum sensory inputs
        for _ in range(self.n_hidden):
            # Code below is straight up stolen from 'activate(self, inputs)'
            with torch.no_grad():
                inputs = torch.tensor([self.initial_readings], dtype=self.dtype)
                output_inputs = self.input_to_output.mm(inputs.t()).t()
                self.activations = self.activation(self.input_to_hidden.mm(inputs.t()).t() +
                                                   self.hidden_to_hidden.mm(self.activations.t()).t() +
                                                   self.hidden_biases)
                output_inputs += self.hidden_to_output.mm(self.activations.t()).t()
                self.outputs = self.activation(output_inputs + self.output_biases)
    
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
            output_inputs = self.input_to_output.mm(inputs.t()).t()
            
            # Denote the impact hidden nodes have on the outputs, if there are hidden nodes
            if self.n_hidden > 0:
                # Nice to know:
                #  - tensor.t() will transpose the tensor
                #  - tensor.mm(tensor2) will perform a matrix multiplication between tensor and tensor2
                
                # The activation is defined by:
                #  - the inputs mapping to the hidden nodes
                #  - the hidden nodes mapping to themselves
                #  - the hidden nodes' biases
                self.activations = self.activation(self.input_to_hidden.mm(inputs.t()).t() +
                                                   self.hidden_to_hidden.mm(self.activations.t()).t() +
                                                   self.hidden_biases)
                output_inputs += self.hidden_to_output.mm(self.activations.t()).t()
            
            # Define the values of the outputs, which is the sum of their received inputs and their corresponding bias
            self.outputs = self.activation(output_inputs + self.output_biases)
        return self.outputs
    
    @staticmethod
    def create(genome,
               config,
               batch_size=1,
               activation=tanh_activation,
               prune_empty=False,
               ):
        """
        This class will unravel the genome and create a feed-forward network based on it. In other words, it will create
        the phenotype (network) suiting the given genome.
        
        :param genome: The genome for which a network must be created
        :param config: Population config
        :param batch_size: Batch-size needed to setup network dimension
        :param activation: Default activation
        :param prune_empty: Remove nodes that do not contribute to final result
        """
        global nonempty
        genome_config = config.genome_config
        
        # Collect the nodes whose state is required to compute the final network output(s), this excludes the inputs
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)
        
        # Prune out the unneeded nodes, which is done by only remaining those nodes that receive a connection, note that
        # the 'union' only makes sure that (all) the inputs are considered as well.
        if prune_empty: nonempty = {conn.key[1] for conn in genome.connections.values() if conn.enabled}.union(
                set(genome_config.input_keys))
        
        # Get a list of all the input, hidden, and output keys
        input_keys = list(genome_config.input_keys)
        hidden_keys = [k for k in genome.nodes.keys() if k not in genome_config.output_keys]
        output_keys = list(genome_config.output_keys)
        
        # Define the biases, note that inputs do not have a bias (since they aren't actually nodes!)
        hidden_biases = [genome.nodes[k].bias for k in hidden_keys]
        output_biases = [genome.nodes[k].bias for k in output_keys]
        
        # Put the biases of the pruned output-nodes to zero
        if prune_empty:
            for i, key in enumerate(output_keys):
                if key not in nonempty:
                    output_biases[i] = 0.0
        
        # Define the input, hidden, and output dimensions (i.e. number of nodes)
        n_inputs = len(input_keys)
        n_hidden = len(hidden_keys)
        n_outputs = len(output_keys)
        
        # Create a mapping of a node's key to their index in their corresponding list
        input_key_to_idx = {k: i for i, k in enumerate(input_keys)}
        hidden_key_to_idx = {k: i for i, k in enumerate(hidden_keys)}
        output_key_to_idx = {k: i for i, k in enumerate(output_keys)}
        
        def key_to_idx(k):
            """Convert key to their corresponding index."""
            return input_key_to_idx[k] if k in input_keys \
                else output_key_to_idx[k] if k in output_keys \
                else hidden_key_to_idx[k]
        
        # Only feed-forward connections considered, these lists contain the connections and their weights respectively
        #  Note that the connections are index-based and not key-based!
        input_to_hidden = ([], [])
        hidden_to_hidden = ([], [])
        input_to_output = ([], [])
        hidden_to_output = ([], [])
        
        # Convert the key-based connections to index-based connections one by one, also save their weights
        for conn in genome.connections.values():
            if not conn.enabled: continue
            
            # Check if connection is necessary
            i_key, o_key = conn.key
            if o_key not in required and i_key not in required: continue
            if prune_empty and i_key not in nonempty:
                print('Pruned {}'.format(conn.key))
                continue
            
            # Convert to index-based
            i_idx = key_to_idx(i_key)
            o_idx = key_to_idx(o_key)
            
            # Store
            if i_key in input_keys and o_key in hidden_keys:
                idxs, vals = input_to_hidden
            elif i_key in hidden_keys and o_key in hidden_keys:
                idxs, vals = hidden_to_hidden
            elif i_key in input_keys and o_key in output_keys:
                idxs, vals = input_to_output
            elif i_key in hidden_keys and o_key in output_keys:
                idxs, vals = hidden_to_output
            else:
                raise ValueError(f'Invalid connection from key {i_key} to key {o_key}')
            
            # Append to the lists of the right tuple
            idxs.append((o_idx, i_idx))  # Connection: to, from
            vals.append(conn.weight)  # Connection: weight
        
        return FeedForwardNet(
                n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs,
                input_to_hidden=input_to_hidden, input_to_output=input_to_output,
                hidden_to_hidden=hidden_to_hidden, hidden_to_output=hidden_to_output,
                hidden_biases=hidden_biases, output_biases=output_biases,
                batch_size=batch_size,
                activation=activation,
        )


def make_net(genome, config, bs=1):
    """
    Create the "brains" of the candidate, based on its genetic wiring.

    :param genome: Genome specifies the brains internals
    :param config: Configuration class
    :param bs: Batch size, which represents amount of games trained in parallel
    """
    return FeedForwardNet.create(genome, config, bs)
