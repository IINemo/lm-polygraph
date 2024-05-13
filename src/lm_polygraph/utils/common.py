import logging
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
    
    return ('sequence', metric_name)
