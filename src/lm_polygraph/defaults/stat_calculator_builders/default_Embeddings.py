from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.stat_calculators.embeddings import (
    EmbeddingsCalculator,
    TokenEmbeddingsCalculator,
)
import logging

log = logging.getLogger("lm_polygraph")


def load_stat_calculator(config, builder):
    if config.embeddings_level == "sequence":
        builder.return_embeddings = True
        return EmbeddingsCalculator()
    elif config.embeddings_level == "token":
        builder.return_token_embeddings = True
        return TokenEmbeddingsCalculator()
    else:
        raise ValueError("Invalid embeddings configuration")
