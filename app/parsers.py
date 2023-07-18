import json
import sys
import numpy as np

from estimators import *

UE_BOUNDS_FILEPATH = 'utils/ue_bounds.json'


def normalize(est: Estimator, val: float) -> float:
    if np.isnan(val):
        return 1
    with open(UE_BOUNDS_FILEPATH, 'r') as f:
        ue_bounds = json.load(f)
    if str(est) not in ue_bounds.keys():
        sys.stderr.write(
            f'Could not find normalizing bounds for estimator: {str(est)}. Will not normalize values.')
        return val
    low, high = ue_bounds[str(est)]['low'], ue_bounds[str(est)]['high']
    print('Normalizing')
    print('  low:', low, 'high:', high, 'cur:', val)
    return min(1, max(0, (high - val) / (high - low)))


def parse_ue_method(method_name: str, model_path: str, cache_path: str) -> Estimator:
    match method_name:
        case "token-level, Maximum Probability":
            return MaxProbabilityToken()
        case "token-level, Normalized Maximum Probability":
            return MaxProbabilityNormalizedToken()
        case "token-level, Entropy":
            return EntropyToken()
        case "token-level, Mutual Information":
            return MutualInformationToken()
        case "token-level, Conditional Mutual Information":
            return ConditionalMutualInformationToken(tau=0.0656, lambd=3.599)
        case "token-level, Attention Entropy":
            return AttentionEntropyToken()
        case "token-level, Attention Recursive Entropy":
            return AttentionRecursiveToken()
        case "token-level, Exponential Attention Entropy":
            return ExponentialAttentionEntropyToken(0.8)
        case "token-level, Exponential Attention Recursive Entropy":
            return ExponentialAttentionEntropyToken(0.8)
        case "token-level, Semantic Entropy":
            return SemanticEntropyToken(model_path, cache_path)
        case "sequence-level, Maximum Probability":
            return MaxProbabilitySeq()
        case "sequence-level, Normalized Maximum Probability":
            return MaxProbabilityNormalizedSeq()
        case "sequence-level, Entropy":
            return EntropySeq()
        case "sequence-level, Mutual Information":
            return MutualInformationSeq()
        case "sequence-level, Conditional Mutual Information":
            return ConditionalMutualInformationSeq(tau=0.0656, lambd=3.599)
        case "sequence-level, Attention Entropy":
            return AttentionEntropySeq()
        case "sequence-level, Attention Recursive Entropy":
            return AttentionRecursiveSeq()
        case "sequence-level, Exponential Attention Entropy":
            return ExponentialAttentionEntropySeq(0.8)
        case "sequence-level, Exponential Attention Recursive Entropy":
            return ExponentialAttentionRecursiveSeq(0.8)
        case "sequence-level, P(True)":
            return PTrue()
        case "sequence-level, P(Uncertainty)":
            return PUncertainty()
        case "sequence-level, Predictive Entropy Sampling":
            return PredictiveEntropy()
        case "sequence-level, Normalized Predictive Entropy Sampling":
            return LengthNormalizedPredictiveEntropy()
        case "sequence-level, Lexical Similarity Rouge-1":
            return LexicalSimilarity(metric='rouge1')
        case "sequence-level, Lexical Similarity Rouge-2":
            return LexicalSimilarity(metric='rouge2')
        case "sequence-level, Lexical Similarity Rouge-L":
            return LexicalSimilarity(metric='rougeL')
        case "sequence-level, Lexical Similarity Rouge-BLEU":
            return LexicalSimilarity(metric='BLEU')
        case "sequence-level, Semantic Entropy":
            return SemanticEntropy()
        case "sequence-level, Adaptive Sampling Predictive Entropy":
            return PredictiveEntropyAdaptedSampling()
        case "sequence-level, Adaptive Sampling Semantic Entropy":
            return SemanticEntropyAdaptedSampling()
        case "sequence-level, Mahalanobis Distance":
            return MahalanobisDistanceSeq()
        case _:
            raise Exception(f'Unknown method: {method_name}')


def parse_model(model: str) -> str:
    match model:
        case 'Dolly 3b':
            return 'databricks/dolly-v2-3b'
        case 'Dolly 7b':
            return 'databricks/dolly-v2-7b'
        case 'Dolly 12b':
            return 'databricks/dolly-v2-12b'
        case 'Bloomz 560M':
            return 'bigscience/bloomz-560m'
        case 'Bloomz 3b':
            return 'bigscience/bloomz-3b'
        case 'Bloomz 7b':
            return 'bigscience/bloomz-7b1'
        case 'Falcon 7b':
            return 'tiiuae/falcon-7b'
        case 'Llama 3b':
            return 'openlm-research/open_llama_3b'
        case 'Llama 7b':
            return 'openlm-research/open_llama_7b'
        case 'Llama 13b':
            return 'openlm-research/open_llama_13b'
        case 'BART Large CNN':
            return 'facebook/bart-large-cnn'
        case 'T5 XL NQ':
            return 'google/t5-xl-ssm-nq'
        case 'Flan T5 XL':
            return 'google/flan-t5-xl'
        case 'OPT 2.7b':
            return 'facebook/opt-2.7b'
        case _:
            raise Exception(f'Unknown model: {model}')
