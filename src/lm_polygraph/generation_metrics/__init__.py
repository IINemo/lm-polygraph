"""Public API, imported on demand to keep optional dependencies optional."""

from lm_polygraph._optional import lazy_exports

_EXPORTS = {
    "RougeMetric": (".rouge", "RougeMetric"),
    "BLEUMetric": (".bleu", "BLEUMetric"),
    "ModelScoreSeqMetric": (".model_score", "ModelScoreSeqMetric"),
    "ModelScoreTokenwiseMetric": (".model_score", "ModelScoreTokenwiseMetric"),
    "BartScoreSeqMetric": (".bart_score", "BartScoreSeqMetric"),
    "AccuracyMetric": (".accuracy", "AccuracyMetric"),
    "AlignScore": (".alignscore", "AlignScore"),
    "OpenAIFactCheck": (".openai_fact_check", "OpenAIFactCheck"),
    "BertScoreMetric": (".bert_score", "BertScoreMetric"),
    "SbertMetric": (".sbert", "SbertMetric"),
    "AggregatedMetric": (".aggregated_metric", "AggregatedMetric"),
    "PreprocessOutputTarget": (".preprocess_output_target", "PreprocessOutputTarget"),
    "Comet": (".comet", "Comet"),
}

__all__ = list(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
