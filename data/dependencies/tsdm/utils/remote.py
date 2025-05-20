r"""Implements a downloader for the TSDM-package."""

__all__ = [
    "hash_file",
    "download",
    "validate_hash",
]

import hashlib
import string
from pathlib import Path
from typing import Any, Optional

import requests
from tqdm.autonotebook import tqdm

from data.dependencies.tsdm.utils.types import PathType


def hash_file(
    file: str, block_size: int = 65536, algorithm: str = "sha256", **kwargs: Any
) -> str:
    r"""Calculate the SHA256-hash of a file."""
    algorithms = vars(hashlib)
    hash_value = algorithms[algorithm](**kwargs)

    with open(file, "rb") as file_handle:
        for byte_block in iter(lambda: file_handle.read(block_size), b""):
            hash_value.update(byte_block)

    return hash_value.hexdigest()


def download(
    url: str, fname: Optional[PathType] = None, chunk_size: int = 1024
) -> None:
    r"""Download a file from a URL."""
    response = requests.get(url, stream=True, timeout=10)
    total = int(response.headers.get("content-length", 0))
    path = Path(fname if fname is not None else url.split("/")[-1])
    try:
        with path.open("wb") as file, tqdm(
            desc=str(path),
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                progress_bar.update(size)
    except Exception as e:
        path.unlink()
        raise RuntimeError(
            f"Error '{e}' occurred while downloading {fname}, deleting partial files."
        ) from e


def validate_hash(fname: str, hash_value: str, hash_type: str = "sha256") -> bool:
    r"""Validate a file against a hash value."""
    return hash_file(fname, algorithm=hash_type) == hash_value


def to_base(n: int, b: int) -> list[int]:
    r"""Convert non-negative integer to any basis.

    References
    ----------
    - https://stackoverflow.com/a/28666223/9318372

    Parameters
    ----------
    n: int
    b: int

    Returns
    -------
    digits: list[int]
        Satisfies: ``n = sum(d*b**k for k, d in enumerate(reversed(digits)))``
    """
    digits = []
    while n:
        n, d = divmod(n, b)
        digits.append(d)
    return digits[::-1] or [0]


def to_alphanumeric(n: int) -> str:
    r"""Convert integer to alphanumeric code."""
    chars = string.ascii_uppercase + string.digits
    digits = to_base(n, len(chars))
    return "".join(chars[i] for i in digits)


# def shorthash(inputs) -> str:
#     r"""Roughly good for 2¹⁶=65536 items."""
#     encoded = json.dumps(dictionary, sort_keys=True).encode()
#
#     return shake_256(inputs).hexdigest(8)
