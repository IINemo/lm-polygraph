from .stat_calculator import StatCalculator, register
from .greedy_probs import GreedyProbsCalculator, BlackboxGreedyTextsCalculator
from .greedy_lm_probs import GreedyLMProbsCalculator
from .prompt import PromptCalculator
from .prompt_avg import PromptCalculatorAVG, prompt_сoh, prompt_flu, prompt_rel, prompt_сon
from .prompt_bb import PromptCalculatorBB, prompt_ue, prompt_quality
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
register(
    PromptCalculatorAVG(
        prompt_сoh,
        [' 1', ' 2', ' 3', ' 4', ' 5'],
        "prompt_coherence",
    )
)
register(
    PromptCalculatorAVG(
        prompt_сon,
        [' 1', ' 2', ' 3', ' 4', ' 5'],
        "prompt_consistency",
    )
)
register(
    PromptCalculatorAVG(
        prompt_rel,
        [' 1', ' 2', ' 3', ' 4', ' 5'],
        "prompt_relevance",
    )
)
register(
    PromptCalculatorAVG(
        prompt_flu,
        [' 1', ' 2', ' 3',],
        "prompt_fluency",
    )
)

register(
    PromptCalculatorBB(
        prompt_ue,
        "prompt_ue",
    )
)

register(
    PromptCalculatorBB(
        prompt_quality,
        "prompt_quality",
    )
)
register(SamplingGenerationCalculator())
register(BlackboxSamplingGenerationCalculator())
register(BartScoreCalculator())
register(ModelScoreCalculator())
register(EmbeddingsCalculator())
register(EnsembleTokenLevelDataCalculator())
register(SemanticMatrixCalculator())
