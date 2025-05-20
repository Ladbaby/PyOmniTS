import hashlib
import json
import os
from typing import Optional

import numpy as np
import scipy.sparse as sp
from layers.tsl import logger
from layers.tsl.datasets import TabularDataset
from layers.tsl.datasets.prototypes.mixin import MissingValuesMixin
from layers.tsl.ops.graph_generators import build_circle_graph, build_knn_graph, \
    build_line_graph
from layers.tsl.ops.imputation import sample_mask
from layers.tsl.utils import files_exist


class GraphMSO(TabularDataset, MissingValuesMixin):
    """The **Graph MSO** (Multiple Superimposed Oscillators) dataset from the
    paper `"Graph-based Forecasting with Missing Data through Spatiotemporal
    Downsampling" <https://arxiv.org/abs/2402.10634>`_ (Marisca et al., 2024).

    The dataset is inspired by a benchmark in time series forecasting [1].
    After creating a graph with a given topology, each node is assigned a
    sinusoid characterized by a frequency that is incommensurable with the ones
    at the other nodes. As such, summing one or more sinusoids results in a
    signal that is aperiodic and, thus, very difficult to predict. In addition,
    by aggregating from neighbors randomly chosen at different hops, predicting
    the signal becomes an even more challenging task.

    To obtain such a signal, we combine each sinusoid with the sinusoids of the
    graph neighbors according to the chosen propagation scheme.

    Args:
        n_nodes (int): Number of nodes in the graph.
        n_steps (int): Number of time steps.
        spatial_order (int, optional): Spatial order for spatial block missing.
            (default: :obj:`1`)
        max_neighbors (int, optional): Maximum number of neighbors for mask
            propagation.
            (default: :obj:`None`)
        graph_generator (str, optional): Graph generator type.
            Options:
                - :obj:`'circle'`: Undirected circle graph.
                - :obj:`'path'`: Undirected path graph with no closed loop.
                - :obj:`'knnX'`: Randomly connect to ``X`` incoming edges,
                  where ``X`` is an integer.
            (default: :obj:`'circle'`)
        p_noise (float, optional): Probability of noise.
            (default: :obj:`0.05`)
        p_fault (float, optional): Probability of fault.
            (default: :obj:`0.01`)
        min_seq (int, optional): Minimum sequence length.
            (default: :obj:`1`)
        max_seq (int, optional): Maximum sequence length.
            (default: :obj:`10`)
        noise (float, optional): Noise level.
            (default: :obj:`0.0`)
        propagate_mask (bool, optional): Flag for spatial block missing.
            (default: :obj:`False`)
        seed (int, optional): Random seed.
            (default: :obj:`None`)
        cached (bool, optional): Flag indicating whether to cache the dataset.
            (default: :obj:`True`)
        root (str, optional): Root directory for the dataset.
            (default: :obj:`None`)

    References:
        [1] Bianchi, F. M., et al., Recurrent neural networks for short-term
        load forecasting: an overview and comparative analysis.
        (Springer Briefs in Computer Science, 2017)
    """
    seed = 64
    version: str = "0.3"

    def __init__(self,
                 n_nodes: int = 100,
                 n_steps: int = 10000,
                 spatial_order: int = 3,
                 max_neighbors: Optional[int] = 5,
                 graph_generator: str = 'knn3',
                 p_noise: float = 0.05,
                 p_fault: float = 0.,
                 min_seq: int = 8,
                 max_seq: int = 48,
                 noise: float = 0.,
                 propagate_mask: bool = False,  # for spatial block missing
                 seed: Optional[int] = 123,
                 cached: bool = True,
                 root: str = None):
        # Save attributes to be compared against cache
        self._n_nodes = n_nodes
        self._n_steps = n_steps
        self.spatial_order = spatial_order
        self.max_neighbors = max_neighbors
        self.graph_generator = graph_generator
        self.p_noise = p_noise
        self.p_fault = p_fault
        self.min_seq = min_seq
        self.max_seq = max_seq
        self.noise = noise
        self.propagate_mask = propagate_mask
        self.root = root

        if seed is None:
            seed = np.random.randint(10e5)
        self.seed = seed

        target, optimal, connectivity, eval_mask = self.load(cached)

        super().__init__(target=target)
        self.add_covariate('optimal', optimal, pattern='t n f')
        self.connectivity = connectivity
        self.set_eval_mask(eval_mask)

    @property
    def required_file_names(self):
        return {'data': 'mso.npz', 'adj_t': 'adj_t.npz',
                'checksum': 'checksum.md5'}

    def clear_cache(self):
        for filename in self.required_files_paths_list:
            if os.path.exists(filename):
                os.unlink(filename)

    def sinusoids(self):
        """
        Returns:
            x (array): ``n_nodes`` sinusoids with incommensurable frequencies.

        Shapes:
            x: (n_nodes, n_steps)
        """
        freq = 1.0
        t = np.arange(self._n_steps * freq, step=freq)
        # Log frequencies
        logger.info(f"{self.__class__.__name__} signal frequencies:")
        min_period = 2 * np.pi * (1 / freq)
        logger.info(f"\tmin period: {min_period:.2f}")
        max_period = np.exp((self._n_nodes - 1) / self._n_nodes) * min_period
        logger.info(f"\tmax period: {max_period:.2f}")
        # Generate sinusoids
        x_t = np.arange(self._n_nodes)
        x = np.sin(1 / np.exp(x_t / self._n_nodes)[:, None] @ t[None])
        return x

    def load_raw(self):
        # Signal
        x = self.sinusoids()

        # Generate edge_index
        rng_adj = np.random.default_rng(self.seed)
        if self.graph_generator == 'circle':
            # randomly permute nodes
            x = rng_adj.permutation(x)
            _, edge_index, _ = build_circle_graph(self._n_nodes,
                                                  undirected=True)
        elif self.graph_generator == 'path':
            # randomly permute nodes
            x = rng_adj.permutation(x)
            _, edge_index, _ = build_line_graph(self._n_nodes)
            # To undirected
            edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
        elif self.graph_generator.startswith('knn'):  # knn4 -> k=4
            k = int(self.graph_generator[3:])
            _, edge_index, _ = build_knn_graph(self._n_nodes, k, rng=rng_adj,
                                               include_self=False)
        else:
            raise NotImplementedError(f"Graph generator {self.graph_generator} "
                                      "not implemented")

        # Binary adj: edge_weights are all 1
        edge_weight = np.ones_like(edge_index[0], dtype=np.float32)

        row, col = edge_index  # edge_index is already A^T
        adj_t = sp.coo_matrix((edge_weight, (row, col)),
                              shape=(self._n_nodes, self._n_nodes))
        adj_t.setdiag(0.)  # remove self-loops
        adj_t.eliminate_zeros()
        kernel = adj_t

        # Propagate signal through multiple hops
        if self.max_neighbors is not None:
            # Compute signal kernel as sum of powers of the adjacency matrix
            kernels = [kernel]  # kernel^1
            for _ in range(self.spatial_order - 1):
                # kernel^1, kernel^2, ...
                last_kernel = kernels[-1]
                # kernel^2, kernel^3, ...
                kernels.append(last_kernel.dot(kernel))
            # kernel^1 + kernel^2 + ... + kernel^spatial_order
            signal_kernel = sum(kernels)
            signal_kernel.setdiag(0)  # remove self-loops
            signal_kernel.eliminate_zeros()  # remove zero entries
            # Set non-zero values to 1
            signal_kernel[signal_kernel.nonzero()] = 1.

            # Sparsify kernel
            matrix = signal_kernel.toarray()
            rng_adj = np.random.default_rng(self.seed + 401)
            for node in range(matrix.shape[0]):
                nnz = matrix[node].nonzero()[0]
                overflow = len(nnz) - self.max_neighbors
                if overflow > 0:
                    # Keep only randomly chosen max_neighbors
                    nnz = rng_adj.choice(nnz, overflow, replace=False)
                    matrix[node, nnz] = 0.
            signal_kernel = sp.csr_matrix(matrix)

            target = x + signal_kernel.dot(x)

            # By using the same kernels, the mask is propagated in the
            # same way as the data is constructed, while we provide a different
            # adjacency matrix as inductive bias.
            kernel = signal_kernel
        # Standard graph propagation
        else:
            target = x
            for _ in range(self.spatial_order):
                target = target + kernel.dot(x)

        # 'n t -> t n f=1'
        target = target.swapaxes(0, 1)[..., None]

        # Add noise
        eta = 0
        if self.noise > 0:
            rng_noise = np.random.default_rng(self.seed + 10)
            eta = rng_noise.normal(scale=self.noise, size=target.shape)

        # Compute evaluation mask
        if self.p_noise > 0 or self.p_fault > 0:
            rng_mask = np.random.default_rng(self.seed + 20)
            eval_mask = sample_mask(target.shape,
                                    p=self.p_fault,
                                    p_noise=self.p_noise,
                                    min_seq=self.min_seq,
                                    max_seq=self.max_seq,
                                    rng=rng_mask)
            # Propagate mask for spatial block missing
            if self.propagate_mask:
                eval_mask_prop = kernel.dot(eval_mask.swapaxes(0, 1)[..., 0])
                eval_mask = eval_mask + eval_mask_prop.swapaxes(0, 1)[..., None]
        else:
            eval_mask = np.zeros_like(target)

        return target + eta, target, adj_t, eval_mask

    def load_cached(self):
        content = np.load(self.required_files_paths['data'])
        target = content['target']
        optimal = content['optimal']
        eval_mask = content['eval_mask']
        connectivity = sp.load_npz(self.required_files_paths['adj_t'])
        return target, optimal, connectivity, eval_mask

    def checksum(self) -> str:
        # Compute checksum
        state = self.__getstate__()
        checksum = hashlib.md5(
            json.dumps(state, sort_keys=True).encode('utf-8')
        ).hexdigest() + f"_v{self.version}"
        return checksum

    def load(self, cached: bool):
        if cached:  # load cached version
            if files_exist(self.required_files_paths_list):
                # compare against cached checksum
                checksum = self.checksum()
                with open(self.required_files_paths['checksum'], 'r') as fp:
                    cached_checksum = fp.read()
                if checksum == cached_checksum:  # checksum matching
                    return self.load_cached()
                else:
                    logger.warning("Cached dataset does not match arguments, "
                                   "re-building the dataset...")
            else:  # cached files do not exist
                logger.info("No cache found, building the dataset...")

        self.clear_cache()  # empty root folder

        # load the data
        target, optimal, connectivity, eval_mask = self.load_raw()

        if cached:
            os.makedirs(self.root_dir, exist_ok=True)
            # save data
            np.savez_compressed(self.required_files_paths['data'],
                                target=target,
                                optimal=optimal,
                                eval_mask=eval_mask)
            # save adj
            sp.save_npz(self.required_files_paths['adj_t'], connectivity)
            # save checksum
            with open(self.required_files_paths['checksum'], 'w') as fp:
                fp.write(self.checksum())

        return target, optimal, connectivity, eval_mask

    def get_connectivity(self,
                         include_self: bool = False,
                         layout: str = 'edge_index',
                         **kwargs):
        if len(kwargs):
            logger.warning(f"{', '.join(kwargs.keys())} arguments ignored in "
                           f"{self.__class__.__name__}.get_connectivity()")
        adj_t = self.connectivity.copy()  # adj_t is a sparse matrix
        if include_self:
            adj_t.setdiag(1.)
        if layout == 'edge_index':
            col, row = adj_t.nonzero()  # edge_index is adj_t
            return np.stack([row, col]).astype(np.int64), adj_t.data
        elif layout == 'dense':
            return adj_t.todense()
        return adj_t
