r"""Base Classes for Encoders."""


from __future__ import annotations

__all__ = [
    # Classes
    "BaseEncoder",
    "IdentityEncoder",
    "ChainedEncoder",
    "ProductEncoder",
    "DuplicateEncoder",
    "CloneEncoder",
]


import logging
from abc import ABC, ABCMeta, abstractmethod
from copy import deepcopy
from typing import Any, ClassVar, Sequence, overload

from data.dependencies.tsdm.utils.decorators import wrap_func
from data.dependencies.tsdm.utils.strings import repr_sequence
from data.dependencies.tsdm.utils.types import ObjectVar


class BaseEncoderMetaClass(ABCMeta):
    r"""Metaclass for BaseDataset."""

    def __init__(cls, *args, **kwargs):
        cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")
        super().__init__(*args, **kwargs)


class BaseEncoder(ABC, metaclass=BaseEncoderMetaClass):
    r"""Base class that all encoders must subclass."""

    LOGGER: ClassVar[logging.Logger]
    r"""Logger for the Encoder."""

    _is_fitted: bool = False
    r"""Whether the encoder has been fitted."""

    def __init__(self) -> None:
        super().__init__()
        self.transform = self.encode
        self.inverse_transform = self.decode

    def __init_subclass__(cls, /, *args: Any, **kwargs: Any) -> None:
        r"""Initialize the subclass.

        The wrapping of fit/encode/decode must be done here to avoid
        `~pickle.PickleError`!
        """
        super().__init_subclass__(*args, **kwargs)
        cls.fit = wrap_func(cls.fit, after=cls._post_fit_hook)  # type: ignore[assignment]
        cls.encode = wrap_func(cls.encode, before=cls._pre_encode_hook)  # type: ignore[assignment]
        cls.decode = wrap_func(cls.decode, before=cls._pre_decode_hook)  # type: ignore[assignment]

    def __matmul__(self, other: BaseEncoder) -> ChainedEncoder:
        r"""Return chained encoders."""
        return ChainedEncoder(self, other)

    def __or__(self, other: BaseEncoder) -> ProductEncoder:
        r"""Return product encoders."""
        return ProductEncoder(self, other)

    def __pow__(self, power: int) -> DuplicateEncoder:
        r"""Return the product encoder of the encoder with itself power many times."""
        return DuplicateEncoder(self, power)

    def __repr__(self) -> str:
        r"""Return a string representation of the encoder."""
        return f"{self.__class__.__name__}()"

    @property
    def is_fitted(self) -> bool:
        r"""Whether the encoder has been fitted."""
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value: bool) -> None:
        self._is_fitted = value

    @property
    def is_surjective(self) -> bool:
        r"""Whether the encoder is surjective."""
        return True

    @property
    def is_injective(self) -> bool:
        r"""Whether the encoder is injective."""
        return True

    @property
    def is_bijective(self) -> bool:
        r"""Whether the encoder is bijective."""
        return self.is_surjective and self.is_injective

    def fit(self, data: Any, /) -> None:
        r"""Implement as necessary."""

    @abstractmethod
    def encode(self, data, /):
        r"""Transform the data."""

    @abstractmethod
    def decode(self, data, /):
        r"""Reverse the applied transformation."""

    def _post_fit_hook(
        self, *args: Any, **kwargs: Any  # pylint: disable=unused-argument
    ) -> None:
        self.is_fitted = True

    def _pre_encode_hook(
        self, *args: Any, **kwargs: Any  # pylint: disable=unused-argument
    ) -> None:
        if not self.is_fitted:
            raise RuntimeError("Encoder has not been fitted.")

    def _pre_decode_hook(
        self, *args: Any, **kwargs: Any  # pylint: disable=unused-argument
    ) -> None:
        if not self.is_fitted:
            raise RuntimeError("Encoder has not been fitted.")


class IdentityEncoder(BaseEncoder):
    r"""Dummy class that performs identity function."""

    def encode(self, data: ObjectVar, /) -> ObjectVar:
        r"""Encode the input."""
        return data

    def decode(self, data: ObjectVar, /) -> ObjectVar:
        r"""Decode the input."""
        return data


class ProductEncoder(BaseEncoder, Sequence[BaseEncoder]):
    r"""Product-Type for Encoders."""

    encoders: list[BaseEncoder]
    r"""The encoders."""

    @property
    def is_fitted(self) -> bool:
        r"""Whether the encoder has been fitted."""
        return all(e.is_fitted for e in self.encoders)

    @is_fitted.setter
    def is_fitted(self, value: bool) -> None:
        for encoder in self.encoders:
            encoder.is_fitted = value

    @property
    def is_surjective(self) -> bool:
        r"""Whether the encoder is surjective."""
        return all(e.is_surjective for e in self.encoders)

    @property
    def is_injective(self) -> bool:
        r"""Whether the encoder is injective."""
        return all(e.is_injective for e in self.encoders)

    def __init__(self, *encoders: BaseEncoder, simplify: bool = True) -> None:
        super().__init__()
        self.encoders = []

        for encoder in encoders:
            if simplify and isinstance(encoder, ProductEncoder):
                for enc in encoder:
                    self.encoders.append(enc)
            else:
                self.encoders.append(encoder)

    def __len__(self) -> int:
        r"""Return the number of the encoders."""
        return len(self.encoders)

    @overload
    def __getitem__(self, index: int) -> BaseEncoder:  # noqa: D105
        ...

    @overload
    def __getitem__(self, index: slice) -> ProductEncoder:  # noqa: D105
        ...

    def __getitem__(self, index: int | slice) -> BaseEncoder | ProductEncoder:
        r"""Get the encoder at the given index."""
        if isinstance(index, int):
            return self.encoders[index]
        if isinstance(index, slice):
            return ProductEncoder(*self.encoders[index])
        raise ValueError(f"Index {index} not supported.")

    def __repr__(self) -> str:
        r"""Pretty print."""
        return repr_sequence(self)

    def fit(self, data: tuple[Any, ...], /) -> None:
        r"""Fit the encoder."""
        for encoder, x in zip(self.encoders, data):
            encoder.fit(x)

    def encode(self, data: tuple[Any, ...], /) -> Sequence[Any]:
        r"""Encode the data."""
        rtype = type(data)
        return rtype(encoder.encode(x) for encoder, x in zip(self.encoders, data))

    def decode(self, data: tuple[Any, ...], /) -> Sequence[Any]:
        r"""Decode the data."""
        rtype = type(data)
        return rtype(encoder.decode(x) for encoder, x in zip(self.encoders, data))


class ChainedEncoder(BaseEncoder, Sequence[BaseEncoder]):
    r"""Represents function composition of encoders."""

    encoders: list[BaseEncoder]
    r"""List of encoders."""

    @property
    def is_fitted(self) -> bool:
        r"""Whether the encoder has been fitted."""
        return all(e.is_fitted for e in self.encoders)

    @is_fitted.setter
    def is_fitted(self, value: bool) -> None:
        for encoder in self.encoders:
            encoder.is_fitted = value

    @property
    def is_surjective(self) -> bool:
        r"""Return True if the encoder is surjective."""
        return all(e.is_surjective for e in self.encoders)

    @property
    def is_injective(self) -> bool:
        r"""Return True if the encoder is injective."""
        return all(e.is_injective for e in self.encoders)

    def __init__(self, *encoders: BaseEncoder, simplify: bool = True) -> None:
        super().__init__()

        self.encoders = []

        for encoder in encoders:
            if simplify and isinstance(encoder, ChainedEncoder):
                for enc in encoder:
                    self.encoders.append(enc)
            else:
                self.encoders.append(encoder)

    def __len__(self) -> int:
        r"""Return number of chained encoders."""
        return len(self.encoders)

    @overload
    def __getitem__(self, index: int) -> BaseEncoder:  # noqa: D105
        ...

    @overload
    def __getitem__(self, index: slice) -> ChainedEncoder:  # noqa: D105
        ...

    def __getitem__(self, index: int | slice) -> BaseEncoder | ChainedEncoder:
        r"""Get the encoder at the given index."""
        if isinstance(index, int):
            return self.encoders[index]
        if isinstance(index, slice):
            return ChainedEncoder(*self.encoders[index])
        raise ValueError(f"Index {index} not supported.")

    def __repr__(self) -> str:
        r"""Pretty print."""
        return repr_sequence(self)

    def fit(self, data: Any, /) -> None:
        r"""Fit to the data."""
        for encoder in reversed(self.encoders):
            encoder.fit(data)
            data = encoder.encode(data)

    def encode(self, data, /):
        r"""Encode the input."""
        for encoder in reversed(self.encoders):
            data = encoder.encode(data)
        return data

    def decode(self, data, /):
        r"""Decode the input."""
        for encoder in self.encoders:
            data = encoder.decode(data)
        return data


class DuplicateEncoder(BaseEncoder):
    r"""Duplicate encoder multiple times (references same object)."""

    def __init__(self, encoder: BaseEncoder, n: int = 1) -> None:
        super().__init__()
        self.base_encoder = encoder
        self.n = n
        self.encoder = ProductEncoder(*(self.base_encoder for _ in range(n)))

        self.is_fitted = self.encoder.is_fitted

    def __repr__(self) -> str:
        r"""Pretty print."""
        return f"Duplicates[{self.n}]@{repr(self.base_encoder)}"

    def fit(self, data: Any, /) -> None:
        r"""Fit the encoder."""
        return self.encoder.fit(data)

    def encode(self, data, /):
        r"""Encode the data."""
        return self.encoder.encode(data)

    def decode(self, data, /):
        r"""Decode the data."""
        return self.encoder.decode(data)


class CloneEncoder(BaseEncoder):
    r"""Clone encoder multiple times (distinct copies)."""

    def __init__(self, encoder: BaseEncoder, n: int = 1) -> None:
        super().__init__()
        self.base_encoder = encoder
        self.n = n
        self.encoder = ProductEncoder(*(deepcopy(self.base_encoder) for _ in range(n)))
        self.is_fitted = self.encoder.is_fitted

    def __repr__(self) -> str:
        r"""Pretty print."""
        return f"Copies[{self.n}]@{repr(self.base_encoder)}"

    def fit(self, data: Any, /) -> None:
        r"""Fit the encoder."""
        return self.encoder.fit(data)

    def encode(self, data, /):
        r"""Encode the data."""
        return self.encoder.encode(data)

    def decode(self, data, /):
        r"""Decode the data."""
        return self.encoder.decode(data)
