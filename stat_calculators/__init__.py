from .greedy_probs import GreedyProbsCalculator
from .greedy_lm_probs import GreedyLMProbsCalculator
from .prompt import PromptCalculator
from .entropy import EntropyCalculator
from .stat_calculator import StatCalculator, register
from .sample import SamplingGenerationCalculator
from .adapted_sample import AdaptedSamplingGenerationCalculator
from .bart_score import BartScoreCalculator
from .model_score import ModelScoreCalculator
from .embeddings import EmbeddingsCalculator

register(GreedyProbsCalculator())
register(EntropyCalculator())
register(GreedyLMProbsCalculator())
register(PromptCalculator('Question: "{q}", answer: "{a}", is the proposed answer correct?', 'True', 'p_true'))
register(PromptCalculator('Do you know the correct answer to the question: "{q}"?', 'Yes', 'p_uncertainty'))
register(SamplingGenerationCalculator())
register(AdaptedSamplingGenerationCalculator())
register(BartScoreCalculator())
register(ModelScoreCalculator())
register(EmbeddingsCalculator())