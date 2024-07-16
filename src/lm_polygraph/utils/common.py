import logging
import importlib.util
import numpy as np
import pandas as pd

from typing import Tuple

log = logging.getLogger("lm_polygraph")


def polygraph_module_init(func):
    def wrapper(*args, **kwargs):
        if func.__name__ == "__init__":
            log.info(f"Initializing {args[0].__class__.__name__}")
        func(*args, **kwargs)

    return wrapper


def seq_man_key(metric_name: str) -> Tuple[str, str]:
    """Convert metric name to format of seq-level name format of
    saved manager archive."""

    return ("sequence", metric_name)


def load_external_module(path_to_file: str):
    """Load external module from file and return it."""

    spec = importlib.util.spec_from_file_location("external_module", path_to_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def is_list_like(obj):
    """Check if object is list-like."""
    valid_types = (list, tuple, set, np.ndarray, pd.Series)
    return isinstance(obj, valid_types)
