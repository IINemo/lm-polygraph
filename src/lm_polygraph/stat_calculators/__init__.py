"""Public API, imported on demand to keep optional dependencies optional."""

from lm_polygraph._optional import lazy_exports

_EXPORTS = {
    "StatCalculator": (".stat_calculator", "StatCalculator"),
    "InitialStateCalculator": (".initial_state", "InitialStateCalculator"),
    "GreedyProbsCalculator": (".greedy_probs", "GreedyProbsCalculator"),
    "BlackboxGreedyTextsCalculator": (
        ".greedy_probs_blackbox",
        "BlackboxGreedyTextsCalculator",
    ),
    "SemanticClassesClaimToSamplesCalculator": (
        ".semantic_classes_claim_to_samples",
        "SemanticClassesClaimToSamplesCalculator",
    ),
    "AttentionForwardPassCalculator": (
        ".attention_forward_pass",
        "AttentionForwardPassCalculator",
    ),
    "GreedyLMProbsCalculator": (".greedy_lm_probs", "GreedyLMProbsCalculator"),
    "GreedyProbsVisualCalculator": (
        ".greedy_visual_probs",
        "GreedyProbsVisualCalculator",
    ),
    "SamplingGenerationVisualCalculator": (
        ".sample_visual",
        "SamplingGenerationVisualCalculator",
    ),
    "GreedyLMProbsVisualCalculator": (
        ".greedy_lm_visual_probs",
        "GreedyLMProbsVisualCalculator",
    ),
    "CrossEncoderSimilarityMatrixVisualCalculator": (
        ".cross_encoder_visual_similarity",
        "CrossEncoderSimilarityMatrixVisualCalculator",
    ),
    "PromptVisualCalculator": (".prompt_visual", "PromptVisualCalculator"),
    "SamplingPromptVisualCalculator": (
        ".prompt_visual",
        "SamplingPromptVisualCalculator",
    ),
    "ClaimPromptVisualCalculator": (".prompt_visual", "ClaimPromptVisualCalculator"),
    "PromptCalculator": (".prompt", "PromptCalculator"),
    "SamplingPromptCalculator": (".prompt", "SamplingPromptCalculator"),
    "ClaimPromptCalculator": (".prompt", "ClaimPromptCalculator"),
    "AttentionElicitingPromptCalculator": (
        ".attention_eliciting_prompt",
        "AttentionElicitingPromptCalculator",
    ),
    "CLAIM_EXTRACTION_PROMPTS": (".claim_level_prompts", "CLAIM_EXTRACTION_PROMPTS"),
    "MATCHING_PROMPTS": (".claim_level_prompts", "MATCHING_PROMPTS"),
    "OPENAI_FACT_CHECK_PROMPTS": (".claim_level_prompts", "OPENAI_FACT_CHECK_PROMPTS"),
    "EntropyCalculator": (".entropy", "EntropyCalculator"),
    "SamplingGenerationCalculator": (".sample", "SamplingGenerationCalculator"),
    "BlackboxSamplingGenerationCalculator": (
        ".sample",
        "BlackboxSamplingGenerationCalculator",
    ),
    "GreedyAlternativesNLICalculator": (
        ".greedy_alternatives_nli",
        "GreedyAlternativesNLICalculator",
    ),
    "GreedyAlternativesFactPrefNLICalculator": (
        ".greedy_alternatives_nli",
        "GreedyAlternativesFactPrefNLICalculator",
    ),
    "BartScoreCalculator": (".bart_score", "BartScoreCalculator"),
    "ModelScoreCalculator": (".model_score", "ModelScoreCalculator"),
    "EmbeddingsCalculator": (".embeddings", "EmbeddingsCalculator"),
    "TrainingStatisticExtractionCalculator": (
        ".statistic_extraction",
        "TrainingStatisticExtractionCalculator",
    ),
    "EnsembleTokenLevelDataCalculator": (
        ".ensemble_token_data",
        "EnsembleTokenLevelDataCalculator",
    ),
    "SemanticMatrixCalculator": (".semantic_matrix", "SemanticMatrixCalculator"),
    "RawInputCalculator": (".raw_input", "RawInputCalculator"),
    "GreedySemanticMatrixCalculator": (
        ".greedy_semantic_matrix",
        "GreedySemanticMatrixCalculator",
    ),
    "ConcatGreedySemanticMatrixCalculator": (
        ".greedy_semantic_matrix",
        "ConcatGreedySemanticMatrixCalculator",
    ),
    "CrossEncoderSimilarityMatrixCalculator": (
        ".cross_encoder_similarity",
        "CrossEncoderSimilarityMatrixCalculator",
    ),
    "GreedyCrossEncoderSimilarityMatrixCalculator": (
        ".greedy_cross_encoder_similarity",
        "GreedyCrossEncoderSimilarityMatrixCalculator",
    ),
    "ClaimsExtractor": (".extract_claims", "ClaimsExtractor"),
    "InferCausalLMCalculator": (
        ".infer_causal_lm_calculator",
        "InferCausalLMCalculator",
    ),
    "SemanticClassesCalculator": (".semantic_classes", "SemanticClassesCalculator"),
    "AttentionForwardPassCalculatorVisual": (
        ".attention_forward_pass_visual",
        "AttentionForwardPassCalculatorVisual",
    ),
    "VLLMLogprobsExtractionCalculator": (
        ".vllm_logprobs_extraction",
        "VLLMLogprobsExtractionCalculator",
    ),
    "SampleSentenceEmbeddingsCalculator": (
        ".sample_sentence_embeddings",
        "SampleSentenceEmbeddingsCalculator",
    ),
}

__all__ = list(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
