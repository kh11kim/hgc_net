"""Pickle-free NPZ wire protocol for the HGC checkpoint service."""

from __future__ import annotations

from io import BytesIO
import json
from typing import Mapping

import numpy as np


PROTOCOL_VERSION = 1


def encode_message(values: Mapping[str, object]) -> bytes:
    """Encode a mapping as NPZ while rejecting object arrays.

    ``np.savez`` itself can silently turn heterogeneous Python values into
    object arrays.  Such arrays would require pickle to decode, so reject them
    before bytes leave the process.
    """

    arrays = {str(name): np.asarray(value) for name, value in values.items()}
    object_arrays = [name for name, value in arrays.items() if value.dtype.hasobject]
    if object_arrays:
        raise TypeError(f"object arrays are not allowed: {object_arrays}")
    stream = BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def decode_message(payload: bytes) -> dict[str, np.ndarray]:
    """Decode NPZ without ever enabling pickle."""

    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def success_response(**values: object) -> bytes:
    return encode_message(
        {
            "protocol_version": np.asarray(PROTOCOL_VERSION, dtype=np.int16),
            "ok": np.asarray(True),
            **values,
        }
    )


def error_response(error: Exception) -> bytes:
    return encode_message(
        {
            "protocol_version": np.asarray(PROTOCOL_VERSION, dtype=np.int16),
            "ok": np.asarray(False),
            "error_type": np.asarray(type(error).__name__),
            "error_message": np.asarray(str(error)),
        }
    )


def scalar(values: Mapping[str, np.ndarray], name: str):
    """Read a scalar field without coercing its dtype."""

    if name not in values:
        raise KeyError(f"request is missing {name}")
    value = values[name]
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar, got shape {value.shape}")
    return value.item()


def json_dict(values: Mapping[str, np.ndarray], name: str) -> dict[str, object]:
    encoded = scalar(values, name)
    if not isinstance(encoded, str):
        raise TypeError(f"{name} must be a JSON string")

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON value {value}")

    decoded = json.loads(encoded, parse_constant=reject_constant)
    if not isinstance(decoded, dict):
        raise TypeError(f"{name} must encode a JSON object")
    return decoded


__all__ = [
    "PROTOCOL_VERSION",
    "decode_message",
    "encode_message",
    "error_response",
    "json_dict",
    "scalar",
    "success_response",
]
