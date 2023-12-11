"""Functions to load the data and retrieve batches."""

from collections.abc import Sequence
import functools
import os
import urllib
import zipfile

import numpy as np


def load_dataset(seq_length: int) -> Sequence[bytes]:
    """Returns an iterator over sequences of bytes, of length seq_length."""
    if not os.path.exists("enwik9"):
        print("Downloading the dataset...")
        # Downloading and extracting the dataset.
        urllib.request.urlretrieve(
            "https://mattmahoney.net/dc/enwik9.zip",
            "enwik9.zip",
        )
        with zipfile.ZipFile("enwik9.zip", "r") as zip_ref:
            zip_ref.extract("enwik9")
        print("Dataset ready to be used!")

    with open("enwik9", "rb") as file:
        all_chunks = list(iter(functools.partial(file.read, seq_length), b""))
    return tuple(all_chunks)


def fetch_random_batch(
    dataset: Sequence[bytes],
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Returns a random batch of shape (batch_size, seq_length)."""
    indexes = rng.choice(len(dataset), size=batch_size)
    batch_list = [np.frombuffer(dataset[i], dtype=np.uint8) for i in indexes]
    return np.array(batch_list, dtype=np.uint8)
