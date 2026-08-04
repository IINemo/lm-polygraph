from lm_polygraph.stat_calculators.sample_sentence_embeddings import (
    SampleSentenceEmbeddingsCalculator,
)
from .utils import load_sentence_embedding_model


def load_stat_calculator(config, builder):
    if not hasattr(builder, "sentence_embedding_model"):
        builder.sentence_embedding_model = load_sentence_embedding_model(
            **config.sentence_embedding_model
        )
    return SampleSentenceEmbeddingsCalculator(builder.sentence_embedding_model)
