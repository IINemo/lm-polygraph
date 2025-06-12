from typing import Tuple
import logging
import importlib.util
from PIL import Image
from transformers import AutoProcessor

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


def flatten_results(results, result_generator_class):
    """
    Flattens a list of lists into a single list.
    Сan be used with any type of result, such as UEs, statistics, or generation metrics.

    Args:
        results: A list of lists, where each sublist contains results for a single input.
                 Expected shape: [num_inputs, num_token_level_results_per_input].
        result_generator_class: The class of the object that generated the results.
                                 Used for error reporting.

    Returns:
        A flattened list of results of shape [num_inputs * num_token_level_results_per_input].

    Raises:
        Exception: If the input is not a list of lists.
    """
    if not isinstance(results, list) or not all(isinstance(x, list) for x in results):
        raise Exception(
            f"Class {result_generator_class} returned {results}, expected list of lists"
        )
    # Flatten the list of lists into a single list
    # The expected shape is [num_inputs, num_token_level_results_per_input]
    return [result for sample_results in results for result in sample_results]


def load_processor(model_path, **kwargs):
    return AutoProcessor.from_pretrained(model_path, **kwargs)


def load_image(image_path):
    return Image.open(image_path).convert("RGB")
