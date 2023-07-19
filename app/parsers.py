import json
import sys

from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, T5ForConditionalGeneration

from estimators import *
from utils.ensemble_generator import EnsembleGenerator
from utils.model import Model

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


def parse_seq_ue_method(method_name: str, model_path: str, cache_path: str) -> Estimator:
    match method_name:
        case "Maximum Probability":
            return MaxProbabilitySeq()
        case "Normalized Maximum Probability":
            return MaxProbabilityNormalizedSeq()
        case "Entropy":
            return EntropySeq()
        case "Mutual Information":
            return MutualInformationSeq()
        case "Conditional Mutual Information":
            return ConditionalMutualInformationSeq(tau=0.0656, lambd=3.599)
        case "Attention Entropy":
            return AttentionEntropySeq()
        case "Attention Recursive Entropy":
            return AttentionRecursiveSeq()
        case "Exponential Attention Entropy":
            return ExponentialAttentionEntropySeq(0.8)
        case "Exponential Attention Recursive Entropy":
            return ExponentialAttentionRecursiveSeq(0.8)
        case "P(True)":
            return PTrue()
        case "P(Uncertainty)":
            return PUncertainty()
        case "Predictive Entropy Sampling":
            return PredictiveEntropy()
        case "Normalized Predictive Entropy Sampling":
            return LengthNormalizedPredictiveEntropy()
        case "Lexical Similarity Rouge-1":
            return LexicalSimilarity(metric='rouge1')
        case "Lexical Similarity Rouge-2":
            return LexicalSimilarity(metric='rouge2')
        case "Lexical Similarity Rouge-L":
            return LexicalSimilarity(metric='rougeL')
        case "Lexical Similarity Rouge-BLEU":
            return LexicalSimilarity(metric='BLEU')
        case "Semantic Entropy":
            return SemanticEntropy()
        case "Adaptive Sampling Predictive Entropy":
            return PredictiveEntropyAdaptedSampling()
        case "Adaptive Sampling Semantic Entropy":
            return SemanticEntropyAdaptedSampling()
        case "EP-T-total-uncertainty":
            return EPTtu()
        case "EP-T-data-uncertainty":
            return EPTdu()
        case "EP-T-mutual-information":
            return EPTmi()
        case "EP-T-rmi":
            return EPTrmi()
        case "EP-T-epkl":
            return EPTepkl()
        case "EP-T-epkl-tu":
            return EPTepkltu()
        case "EP-T-entropy-top5":
            return EPTent5()
        case "EP-T-entropy-top10":
            return EPTent10()
        case "EP-T-entropy-top15":
            return EPTent15()
        case "PE-T-total-uncertainty":
            return PETtu()
        case "PE-T-data-uncertainty":
            return PETdu()
        case "PE-T-mutual-information":
            return PETmi()
        case "PE-T-rmi":
            return PETrmi()
        case "PE-T-epkl":
            return PETepkl()
        case "PE-T-epkl-tu":
            return PETepkltu()
        case "PE-T-entropy-top5":
            return PETent5()
        case "PE-T-entropy-top10":
            return PETent10()
        case "PE-T-entropy-top15":
            return PETent15()
        case "EP-S-total-uncertainty":
            return EPStu()
        case "EP-S-rmi":
            return EPSrmi()
        case "EP-S-rmi-abs":
            return EPSrmiabs()
        case "PE-S-total-uncertainty":
            return PEStu()
        case "PE-S-rmi":
            return PESrmi()
        case "PE-S-rmi-abs":
            return PESrmiabs()
        case _:
            raise Exception(f'Unknown method: {method_name}')


def parse_tok_ue_method(method_name: str, model_path: str, cache_path: str) -> Estimator:
    match method_name:
        case "Maximum Probability":
            return MaxProbabilityToken()
        case "Normalized Maximum Probability":
            return MaxProbabilityNormalizedToken()
        case "Entropy":
            return EntropyToken()
        case "Mutual Information":
            return MutualInformationToken()
        case "Conditional Mutual Information":
            return ConditionalMutualInformationToken(tau=0.0656, lambd=3.599)
        case "Attention Entropy":
            return AttentionEntropyToken()
        case "Attention Recursive Entropy":
            return AttentionRecursiveToken()
        case "Exponential Attention Entropy":
            return ExponentialAttentionEntropyToken(0.8)
        case "Exponential Attention Recursive Entropy":
            return ExponentialAttentionEntropyToken(0.8)
        case "Semantic Entropy":
            return SemanticEntropyToken(model_path, cache_path)
        case _:
            raise Exception(f'Unknown method: {method_name}')


def parse_model(model: str) -> str:
    match model:
        case "Dolly 3b":
            return 'databricks/dolly-v2-3b'
        case "Dolly 7b":
            return 'databricks/dolly-v2-7b'
        case "Dolly 12b":
            return 'databricks/dolly-v2-12b'
        case "Bloomz 560M":
            return 'bigscience/bloomz-560m'
        case "Bloomz 3b":
            return 'bigscience/bloomz-3b'
        case "Bloomz 7b":
            return 'bigscience/bloomz-7b1'
        case "Falcon 7b":
            return 'tiiuae/falcon-7b'
        case "Llama 3b":
            return 'openlm-research/open_llama_3b'
        case "Llama 7b":
            return 'openlm-research/open_llama_7b'
        case "Llama 13b":
            return 'openlm-research/open_llama_13b'
        case _:
            raise Exception(f'Unknown model: {model}')


def parse_ensemble(path: str) -> EnsembleGenerator:
    path = Path(path) 

    model_paths = [model_dir for model_dir in path.iterdir()]
    
    # TODO: implement devices for models
    devices = ['cpu'] * (len(model_paths) - 1)

    model = Model.from_pretrained(model_paths[0])

    ensemble_model = EnsembleGenerator.from_pretrained(model_paths[0]).eval()
    models = [T5ForConditionalGeneration.from_pretrained(path).eval() for path in model_paths[1:]]
    ensemble_model.add_ensemble_models(models, devices)
    
    ensemble_model.tokenizer = AutoTokenizer.from_pretrained(model_paths[0], padding_side="left", add_bos_token=True, model_max_length=256)

    return model, ensemble_model
