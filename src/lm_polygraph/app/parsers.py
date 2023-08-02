import json
import sys

import os
from pathlib import Path
from typing import Tuple

import numpy as np

from lm_polygraph.estimators import *
from lm_polygraph.utils.ensemble_generator import EnsembleGenerationMixin
from lm_polygraph.utils.model import WhiteboxModel


def parse_seq_ue_method(method_name: str, model_path: str, cache_path: str) -> Estimator:
    dataset_name = "triviaqa"
    model_name = model_path.split('/')[-1]
    density_based_ue_params_path = f"/home/jovyan/projects/lm-polygraph/workdir/{dataset_name}/{model_name}"    
    match method_name:
        case "Maximum Probability":
            return MaxProbabilitySeq()
        case "Normalized Maximum Probability":
            return MaxProbabilityNormalizedSeq()
        case "Perplexity":
            return PerplexitySeq()
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
        case "Eigenvalue Laplacian":
            return EigValLaplacian()
        case "Eccentricity":
            return Eccentricity()
        case "Degree Matrix":
            return DegMat()
        case "Number of Semantic Sets":
            return NumSemSets()
        case "Semantic Entropy":
            return SemanticEntropy()
        case "Adaptive Sampling Predictive Entropy":
            return PredictiveEntropyAdaptedSampling()
        case "Adaptive Sampling Semantic Entropy":
            return SemanticEntropyAdaptedSampling()
        case "Mahalanobis Distance":
            return MahalanobisDistanceSeq("decoder", parameters_path=density_based_ue_params_path, normalize=True)
        case "Mahalanobis Distance - Encoder":
            return MahalanobisDistanceSeq("encoder", parameters_path=density_based_ue_params_path, normalize=True)
        case "RDE":
            return RDESeq("decoder", parameters_path=density_based_ue_params_path, normalize=True)
        case "RDE - Encoder":
            return RDESeq("encoder", parameters_path=density_based_ue_params_path, normalize=True)  
        case "PPL+MD":
            return PPLMDSeq("decoder", md_type="MD", parameters_path=density_based_ue_params_path)
        case "PPL+MD - Encoder":
            return PPLMDSeq("encoder", md_type="MD", parameters_path=density_based_ue_params_path)
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
        case "gpt-4":
            return 'openai-gpt-4'
        case "gpt-3.5-turbo":
            return 'openai-gpt-3.5-turbo'
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

def parse_ensemble(path: str, device: str = 'cpu') -> Tuple[WhiteboxModel, WhiteboxModel]:
    if os.path.exists(path):
        path = Path(path) 
        model_paths = [model_dir for model_dir in path.iterdir()]
    else:
        model_paths = path.split(',')

    # TODO: implement devices for models
    devices = [device] * (len(model_paths) - 1)

    model = WhiteboxModel.from_pretrained(model_paths[0])
    ensemble_model = WhiteboxModel.from_pretrained(model_paths[0])
    
    ensemble_model.model.__class__ = type('EnsembleModel',
                                          (ensemble_model.model.__class__,
                                           EnsembleGenerationMixin),
                                          {}) 

    models = [WhiteboxModel.from_pretrained(path).model for path in model_paths[1:]]
    ensemble_model.model.add_ensemble_models(models, devices)

    return model, ensemble_model
