from typing import Tuple, Union

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from layers.tsl.datasets import TabularDataset
from layers.tsl.ops.connectivity import parse_connectivity
from layers.tsl.ops.imputation import to_missing_values_dataset
from layers.tsl.typing import SparseTensArray, DataArray
from layers.tsl.utils import ensure_list

from layers.HD_TTS.lib.pooling import connectivity_to_adj_t


def sample_mask(shape,
                p_fault: float,
                p_noise: float,
                min_seq: int,
                max_seq: int,
                p_propagation: float = 0.,
                edge_index: Adj = None,
                propagation_hops: int = 1,
                dim: int = 1,
                rng: torch.Generator = None,
                device: torch.device = None) -> Tensor:
    # e.g., 1) shape = (num_samples, window, n_nodes, n_channels)
    #    or 2) shape = (n_steps, n_nodes, n_channels)
    if rng is None:
        rng = torch.Generator()

    ndim = len(shape)
    if dim < 0:
        dim = ndim - dim

    # Initialize p_fault mask
    mask = torch.lt(torch.rand(shape, generator=rng, device=device), p_fault)

    # Get indices of faults' beginning
    mask_idx = torch.nonzero(mask)  # shape: [n_faults, number of mask dims]
    # Sample len for each fault
    lens = torch.randint(min_seq, max_seq + 1, (len(mask_idx),),
                         generator=rng, device=device)

    # Propagate faults through the graph
    if edge_index is not None and propagation_hops > 0:

        # Initialize propagation probabilities
        # (can propagate with different p for each hop)
        p_propagation = ensure_list(p_propagation)
        while len(p_propagation) < propagation_hops:
            p_propagation = p_propagation[:1] + p_propagation
        assert len(p_propagation) == propagation_hops
        # Compute actual propagation probability for each hop
        p_propagation = (torch.tensor(p_propagation, dtype=torch.float32)
                         .flip(0)
                         .diff(prepend=torch.zeros(1, dtype=torch.float32))
                         .flip(0)
                         .tolist())

        # Initialize adjacency matrix
        adj = connectivity_to_adj_t(edge_index).t()
        adj_k = adj = adj.set_diag(1)  # Add self-loops to not lose src nodes
        # Initialize propagation mask lists
        prop_faults = []
        prop_lens = []

        # Sample faults to propagate
        for p_prop in p_propagation:

            if p_prop == 0:
                continue

            # Sample fault to propagate
            to_propagate = torch.lt(torch.rand(len(mask_idx), generator=rng,
                                               device=device), p_prop)
            # Separate propagated faults from not propagated ones
            prop_mask_idx = mask_idx[to_propagate]
            mask_idx = mask_idx[~to_propagate]

            # Get src and tgt nodes for each propagated fault
            src_nodes = prop_mask_idx[:, dim + 1]  # Assume node right after dim
            # Compute degree of each src node
            deg = adj_k.fill_value(1).sum(1)[src_nodes]
            _, tgt_nodes, _ = adj_k[src_nodes].coo()  # finally get tgt nodes
            # Add an entry for each fault propagated to each neighbor
            prop_mask_idx = torch.repeat_interleave(prop_mask_idx, deg, dim=0)
            # Update node index with tgt nodes (was src nodes for repetition)
            prop_mask_idx[:, dim + 1] = tgt_nodes
            prop_faults.append(prop_mask_idx)

            # Update lens with repetitions
            new_lens = torch.repeat_interleave(lens[to_propagate], deg, dim=0)
            prop_lens.append(new_lens)
            lens = lens[~to_propagate]

            # Update adj for next hop
            adj_k = adj_k @ adj
        # Concatenate propagated and not propagated faults
        lens = torch.cat([lens] + prop_lens)
        mask_idx = torch.cat([mask_idx] + prop_faults)

    n_faults = len(mask_idx)
    # Make sure faults do not overlap
    unmask_idx = mask_idx.clone()
    unmask_idx[:, dim] += lens
    # Add an entry for each step of the faults in mask_idxs
    mask_idx = torch.repeat_interleave(mask_idx, lens, dim=0)
    # For each added entry, increment the step counter along dim
    increments = torch.arange(max_seq, device=device)[None]
    increments = torch.tile(increments, (n_faults, 1))
    increments = increments[increments < lens[:, None]]
    # increments example:  [0, 1, 2, 0, 1, 0, 1, 2, 3, 4, ...]
    # eventually add increments: now mask_idx has an entry for each masked step
    mask_idx[:, dim] += increments

    # Remove out-of-bounds indices (faults lasting longer than the sequence)
    mask_idx = mask_idx[mask_idx[:, dim] < shape[dim]]
    # Now set values to True in the original (dense) mask
    mask[tuple(mask_idx.T)] = True

    # Add noise
    noise = torch.lt(torch.rand(shape, generator=rng, device=device), p_noise)
    mask = torch.logical_or(mask, noise)

    # Remove out-of-bounds indices to be unmasked
    unmask_idx = unmask_idx[unmask_idx[:, dim] < shape[dim]]
    # Now set values to False in the original (dense) mask
    mask[tuple(unmask_idx.T)] = False

    return mask


def add_missing_values(dataset: TabularDataset,
                       p_fault: float,
                       p_noise: float,
                       min_seq: int,
                       max_seq: int,
                       p_propagation: float = 0.,
                       connectivity: Union[SparseTensArray,
                       Tuple[DataArray]] = None,
                       propagation_hops: int = 1,
                       seed: int = None,
                       inplace: bool = True):
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    connectivity = parse_connectivity(connectivity=connectivity,
                                      target_layout='sparse',
                                      num_nodes=dataset.n_nodes)

    # Compute evaluation mask
    shape = (dataset.length, dataset.n_nodes, dataset.n_channels)
    eval_mask = sample_mask(shape,
                            p_fault=p_fault,
                            p_noise=p_noise,
                            min_seq=min_seq,
                            max_seq=max_seq,
                            p_propagation=p_propagation,
                            edge_index=connectivity,
                            propagation_hops=propagation_hops,
                            dim=0,
                            rng=rng).numpy()

    # Convert to missing values dataset
    dataset = to_missing_values_dataset(dataset, eval_mask, inplace)

    # Store evaluation mask params in dataset
    dataset.p_fault = p_fault
    dataset.p_noise = p_noise
    dataset.min_seq = min_seq
    dataset.max_seq = max_seq
    dataset.p_propagation = p_propagation
    dataset.propagation_hops = propagation_hops
    dataset.seed = seed

    return dataset
