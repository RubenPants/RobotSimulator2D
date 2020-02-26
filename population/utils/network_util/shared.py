"""
shared.py

Shared network functions.
"""
import numpy as np
import torch


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
