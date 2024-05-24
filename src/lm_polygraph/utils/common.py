from typing import Tuple


def seq_man_key(metric_name: str) -> Tuple[str, str]:
    """Convert metric name to format of seq-level name format of
    saved manager archive."""
    return ("sequence", metric_name)
