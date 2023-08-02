from .stat_calculator import StatCalculator, register
from .greedy_probs import GreedyProbsCalculator
from .greedy_lm_probs import GreedyLMProbsCalculator
from .prompt import PromptCalculator
from .entropy import EntropyCalculator
from .std import StdCalculator
from .sample import SamplingGenerationCalculator
from .adapted_sample import AdaptedSamplingGenerationCalculator
from .bart_score import BartScoreCalculator
from .model_score import ModelScoreCalculator
from .embeddings import EmbeddingsCalculator
from .ensemble_token_data import EnsembleTokenLevelDataCalculator

register(GreedyProbsCalculator())
register(EntropyCalculator())
register(StdCalculator())
register(GreedyLMProbsCalculator())
register(PromptCalculator(
    'Question: {q}\n Here are some ideas that were brainstormed: {s}\n Possible answer:{a}\n '
    'Is the possible answer:\n (A) True\n (B) False\n The possible answer is:', 'True', 'p_true'))
register(PromptCalculator(
    'Question: {q}\n Here are some ideas that were brainstormed: {s}\n'
    'Do you know what is the correct answer to the question?\n (A) Yes\n (B) No\n', 'Yes', 'p_uncertainty'))
register(SamplingGenerationCalculator())
register(AdaptedSamplingGenerationCalculator())
register(BartScoreCalculator())
register(ModelScoreCalculator())
register(EmbeddingsCalculator())
register(EnsembleTokenLevelDataCalculator())