"""Load the static Justin q_open from the KMK gripper template."""

from __future__ import annotations

from pathlib import Path

import numpy as np


TEMPLATE_NAMES = ("finger2", "finger3", "finger4")


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
