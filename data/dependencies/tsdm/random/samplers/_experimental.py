r"""Unused code."""

# class SequenceSampler(Sampler):
#     r"""Samples sequences of length seq_len."""
#
#     data: Sized
#     r"""The dataset."""
#     idx: NDArray
#     r"""A list of all valid starting indices."""
#     seq_len: int
#     r"""The static sequence length."""
#     shuffle: bool
#     r"""Whether to sample in random order."""
#
#     def __init__(self, data_source: Sized, /, *, seq_len: int, shuffle: bool = True):
#         r"""Initialize the Sampler.
#
#         Parameters
#         ----------
#         data_source: Sized
#         seq_len: int
#         shuffle: bool = True
#         """
#         super().__init__(data_source)
#         self.data = data_source
#         self.seq_len = seq_len
#         self.idx = np.arange(len(self.data) - self.seq_len)
#         self.shuffle = shuffle
#
#     def __len__(self):
#         r"""Return the maximum allowed index."""
#         return len(self.idx)
#
#     def __iter__(self):
#         r"""Return Indices of the Samples."""
#         indices = self.idx[permutation(len(self))] if self.shuffle else self.idx
#
#         for i in indices:
#             yield np.arange(i, i + self.seq_len)


# class CollectionSampler(Sampler):
#     r"""Samples a single random  object from."""
#
#     def __init__(self, data_source: Sized, shuffle: bool = True):
#         super().__init__(data_source)
#         self.data = data_source
#         self.shuffle = shuffle
#         assert hasattr(data_source, "index"), "Data must have index."
#         assert isinstance(data_source.index, Index), "Index must be `pandas.Index`."
#         self.idx = data_source.index
#
#     def __len__(self):
#         r"""Return the maximum allowed index."""
#         return len(self.idx)
#
#     def __iter__(self):
#         r"""Return Indices of the Samples."""
#         indices = self.idx[permutation(len(self))] if self.shuffle else self.idx
#
#         for i in indices:
#             yield i


# class MappingSampler(Sampler):
#     r"""Sample from a Mapping of Datasets.
#
#     To be used in conjunction with `tsdm.datasets.torch.MappingDataset`.
#     """
#
#     def __init__(self, data_source: Mapping[Any, TorchDataset], shuffle: bool = True):
#         super().__init__(data_source)
#         self.data = data_source
#         self.shuffle = shuffle
#         self.index = list(data_source.keys())
#
#     def __len__(self) -> int:
#         r"""Return the maximum allowed index."""
#         return len(self.data)
#
#     def __iter__(self) -> Iterator[TorchDataset]:
#         r"""Sample from the dataset."""
#         if self.shuffle:
#             perm = np.random.permutation(self.index)
#         else:
#             perm = self.index
#
#         for k in perm:
#             yield self.data[k]


# class BatchSampler(Sampler[list[int]]):
#     r"""Wraps another sampler to yield a mini-batch of indices.
#
#     Args:
#         sampler (Sampler or Iterable): Base sampler. Can be any iterable object
#         batch_size (int): Size of mini-batch.
#         drop_last (bool): If `True`, the sampler will drop the last batch if
#             its size would be less than `batch_size`
#
#     Example:
#         >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
#         [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
#         >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
#         [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
#     """
#
#     def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None:
#         # Since collections.abc.Iterable does not check for `__getitem__`, which
#         # is one way for an object to be an iterable, we don't do an `isinstance`
#         # check here.
#         super().__init__(sampler)
#         if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
#                 batch_size <= 0:
#             raise ValueError("batch_size should be a positive integer value, "
#                              "but got batch_size={}".format(batch_size))
#         if not isinstance(drop_last, bool):
#             raise ValueError("drop_last should be a boolean value, but got "
#                              "drop_last={}".format(drop_last))
#         self.sampler = sampler
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#
#     def __iter__(self) -> Iterator[list[int]]:
#         batch = []
#         for idx in self.sampler:
#             batch.append(idx)
#             if len(batch) == self.batch_size:
#                 yield batch
#                 batch = []
#         if len(batch) > 0 and not self.drop_last:
#             yield batch
#
#     def __len__(self) -> int:
#         # Can only be called if self.sampler has __len__ implemented
#         # We cannot enforce this condition, so we turn off typechecking for the
#         # implementation below.
#         # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
#         if self.drop_last:
#             return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
#         else:
#             return (len(self.sampler) + self.batch_size - 1) // self.batch_size
