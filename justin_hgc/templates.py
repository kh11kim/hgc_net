"""Load the static Justin q_open from the KMK gripper template."""

from __future__ import annotations

from pathlib import Path

import numpy as np


TEMPLATE_NAMES = ("finger2", "finger3", "finger4")

# The server intentionally owns this small immutable runtime datum.  A
# checkpoint therefore remains the only startup artifact: callers do not need
# to provide a KMK YAML file just to decode the model's template index.  Keep
# the values in the same joint order as the canonical Justin-right template
# and use ``validate_canonical_template_q_open`` when checking an external
# template file.
CANONICAL_Q_OPEN = np.asarray(
    (
        (-0.034, 0.0, 0.0, 0.0, 0.415, 0.0, 0.0, 0.0, 1.5707963267948966, 0.0, 0.0, 1.5707963267948966),
        (-0.194, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.5707963267948966),
        (-0.354, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4, 0.0),
    ),
    dtype=np.float32,
)


def load_template_q_open(path: str | Path, names: tuple[str, ...] = TEMPLATE_NAMES) -> np.ndarray:
    """Read q_open, in joint_order, from the selected Justin KMK YAML template."""
    import yaml

    path = Path(path)
    with path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    templates = config.get("grasp_templates", {})
    actual_names = tuple(templates)
    if actual_names != names:
        raise ValueError(
            f"{path}: canonical grasp_type_idx order must be {names}, found {actual_names}"
        )
    result = []
    for name in names:
        try:
            q_open = np.asarray(templates[name]["q_open"], dtype=np.float32)
        except KeyError as error:
            raise KeyError(f"{path}: missing KMK grasp_templates.{name}.q_open") from error
        if q_open.shape != (12,):
            raise ValueError(f"{path}: template {name} q_open must have 12 values")
        result.append(q_open)
    return np.stack(result, axis=0)


def canonical_template_q_open() -> np.ndarray:
    """Return a copy of the canonical ``finger2, finger3, finger4`` values."""

    return CANONICAL_Q_OPEN.copy()


def validate_canonical_template_q_open(
    values: np.ndarray, *, atol: float = 1.0e-7
) -> np.ndarray:
    """Validate and return template values against the canonical runtime datum."""

    actual = np.asarray(values)
    if actual.shape != CANONICAL_Q_OPEN.shape:
        raise ValueError(
            "canonical Justin q_open must have shape "
            f"{CANONICAL_Q_OPEN.shape}, got {actual.shape}"
        )
    if actual.dtype.hasobject or not np.issubdtype(actual.dtype, np.number):
        raise TypeError("canonical Justin q_open must be numeric")
    if not np.isfinite(actual).all():
        raise ValueError("canonical Justin q_open must be finite")
    if not np.allclose(actual, CANONICAL_Q_OPEN, rtol=0.0, atol=float(atol)):
        raise ValueError("Justin q_open does not match the canonical template values")
    return actual.astype(np.float32, copy=True)


__all__ = [
    "CANONICAL_Q_OPEN",
    "TEMPLATE_NAMES",
    "canonical_template_q_open",
    "load_template_q_open",
    "validate_canonical_template_q_open",
]
