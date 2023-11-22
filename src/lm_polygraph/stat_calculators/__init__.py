from .stat_calculator import StatCalculator, register
from .greedy_probs import GreedyProbsCalculator, BlackboxGreedyTextsCalculator
from .greedy_lm_probs import GreedyLMProbsCalculator
from .prompt import PromptCalculator
from .entropy import EntropyCalculator
from .sample import SamplingGenerationCalculator, BlackboxSamplingGenerationCalculator
from .bart_score import BartScoreCalculator
from .model_score import ModelScoreCalculator
from .embeddings import EmbeddingsCalculator
from .ensemble_token_data import EnsembleTokenLevelDataCalculator
from .semantic_matrix import SemanticMatrixCalculator

register(GreedyProbsCalculator())
register(BlackboxGreedyTextsCalculator())
register(EntropyCalculator())
register(GreedyLMProbsCalculator())
register(
    PromptCalculator(
        "Question: {q}\n Possible answer:{a}\n "
        "Is the possible answer:\n (A) True\n (B) False\n The possible answer is:",
        "True",
        "p_true",
    )
)
register(
    PromptCalculator(
        "Question: {q}\n Here are some ideas that were brainstormed: {s}\n Possible answer:{a}\n "
        "Is the possible answer:\n (A) True\n (B) False\n The possible answer is:",
        "True",
        "p_true_sampling",
    )
)
register(SamplingGenerationCalculator())
register(BlackboxSamplingGenerationCalculator())
register(BartScoreCalculator())
register(ModelScoreCalculator())
register(EmbeddingsCalculator())
register(EnsembleTokenLevelDataCalculator())
register(SemanticMatrixCalculator())
