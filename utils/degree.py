from typing import Optional

import tensorlayerx as tlx

from .num_nodes import maybe_num_nodes


def degree(index, num_nodes: Optional[int] = None, dtype=None):
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Parameters
    ----------
    index: tensor
        Index tensor.
    num_nodes: int, optional
        The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    dtype: :obj:`tlx.dtype`, optional
        The desired data type of the
        returned tensor.

    Returns
    -------
    :class:`Tensor`

    """
    N = maybe_num_nodes(index, num_nodes)
    if dtype is None:
        out = tlx.zeros((N, ))
    else:
        out = tlx.zeros((N,), dtype=dtype)
    one = tlx.ones((index.shape[0], ), dtype=out.dtype)
    # out = out.to(N.device)
    # one = one.to(N.device)
    return tlx.unsorted_segment_sum(one, index, N)


# def degree(index, num_nodes: Optional[int] = None, dtype=None):
#     device = index.device
#     N = tlx.convert_to_tensor(num_nodes, dtype=tlx.int32, device=device)
#     if dtype is None:
#         out = tlx.zeros((N,), device=device)
#     else:
#         out = tlx.zeros((N,), dtype=dtype, device=device)
#     one = tlx.ones((index.shape[0],), dtype=out.dtype, device=device)
#     mm = tlx.unsorted_segment_sum(one, index, N)
#     mm = mm.to(device)
#     return mm