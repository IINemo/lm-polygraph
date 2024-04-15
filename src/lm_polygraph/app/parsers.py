import os
from pathlib import Path
from typing import Tuple

from lm_polygraph.estimators import *
from lm_polygraph.utils.model import WhiteboxModel, create_ensemble


def parse_seq_ue_method(
    method_name: str, model_path: str, cache_path: str
) -> Estimator:
    dataset_name = "triviaqa"
    model_name = model_path.split("/")[-1]
    density_based_ue_params_path = (
        f"/home/jovyan/projects/lm-polygraph/workdir/{dataset_name}/{model_name}"
    )
    match method_name:
        case "Maximum Sequence Probability":
            return MaximumSequenceProbability()
        case "Perplexity":
            return Perplexity()
        case "Mean Token Entropy":
            return MeanTokenEntropy()
        case "Mean Pointwise Mutual Information":
            return MeanPointwiseMutualInformation()
        case "Mean Conditional Pointwise Mutual Information":
            return MeanConditionalPointwiseMutualInformation()
        case "P(True)":
            return PTrue()
        case "P(True) Sampling":
            return PTrueSampling()
        case "Monte Carlo Sequence Entropy":
            return MonteCarloSequenceEntropy()
        case "Monte Carlo Normalized Sequence Entropy":
            return MonteCarloNormalizedSequenceEntropy()
        case "Lexical Similarity":
            return LexicalSimilarity()
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
        case "Mahalanobis Distance":
            return MahalanobisDistanceSeq(
                "decoder", parameters_path=density_based_ue_params_path, normalize=True
            )
        case "Mahalanobis Distance - Encoder":
            return MahalanobisDistanceSeq(
                "encoder", parameters_path=density_based_ue_params_path, normalize=True
            )
        case "RDE":
            return RDESeq(
                "decoder", parameters_path=density_based_ue_params_path, normalize=True
            )
        case "RDE - Encoder":
            return RDESeq(
                "encoder", parameters_path=density_based_ue_params_path, normalize=True
            )
        case "HUQ - Decoder":
            return PPLMDSeq(
                "decoder", md_type="MD", parameters_path=density_based_ue_params_path
            )
        case "HUQ - Encoder":
            return PPLMDSeq(
                "encoder", md_type="MD", parameters_path=density_based_ue_params_path
            )
        case "EP-T-Total-Uncertainty":
            return EPTtu()
        case "EP-T-Data-Uncertainty":
            return EPTdu()
        case "EP-T-Mutual-Information":
            return EPTmi()
        case "EP-T-RMI":
            return EPTrmi()
        case "EP-T-EPKL":
            return EPTepkl()
        case "EP-T-Entropy-Top5":
            return EPTent5()
        case "EP-T-Entropy-Top10":
            return EPTent10()
        case "EP-T-Entropy-Top15":
            return EPTent15()
        case "PE-T-Total-Uncertainty":
            return PETtu()
        case "PE-T-Data-Uncertainty":
            return PETdu()
        case "PE-T-Mutual-Information":
            return PETmi()
        case "PE-T-RMI":
            return PETrmi()
        case "PE-T-EPKL":
            return PETepkl()
        case "PE-T-Entropy-Top5":
            return PETent5()
        case "PE-T-Entropy-Top10":
            return PETent10()
        case "PE-T-Entropy-Top15":
            return PETent15()
        case "EP-S-Total-Uncertainty":
            return EPStu()
        case "EP-S-RMI":
            return EPSrmi()
        case "PE-S-Total-Uncertainty":
            return PEStu()
        case "PE-S-RMI":
            return PESrmi()
        case _:
            raise Exception(f"Unknown method: {method_name}")


def parse_tok_ue_method(
    method_name: str, model_path: str, cache_path: str
) -> Estimator:
    match method_name:
        case "Maximum Token Probability":
            return MaximumTokenProbability()
        case "Token Entropy":
            return TokenEntropy()
        case "Pointwise Mutual Information":
            return PointwiseMutualInformation()
        case "Conditional Pointwise Mutual Information":
            return ConditionalPointwiseMutualInformation(tau=0.0656, lambd=3.599)
        case "Semantic Token Entropy":
            return SemanticEntropyToken(model_path, cache_path)
        case _:
            raise Exception(f"Unknown method: {method_name}")


def parse_model(model: str) -> str:
    match model:
        case "GPT-4":
            return "openai-gpt-4"
        case "GPT-3.5-turbo":
            return "openai-gpt-3.5-turbo"
        case "Dolly 3b":
            return "databricks/dolly-v2-3b"
        case "Dolly 7b":
            return "databricks/dolly-v2-7b"
        case "Dolly 12b":
            return "databricks/dolly-v2-12b"
        case "BLOOMz 560M":
            return "bigscience/bloomz-560m"
        case "BLOOMz 3b":
            return "bigscience/bloomz-3b"
        case "BLOOMz 7b":
            return "bigscience/bloomz-7b1"
        case "Falcon 7b":
            return "tiiuae/falcon-7b"
        case "Llama 2 7b":
            return "meta-llama/Llama-2-7b-chat-hf"
        case "Llama 2 13b":
            return "meta-llama/Llama-2-13b-chat-hf"
        case "Vicuna 7b":
            return "lmsys/vicuna-7b-v1.5"
        case "Vicuna 13b":
            return "lmsys/vicuna-13b-v1.5"
        case "Open Llama 3b":
            return "openlm-research/open_llama_3b"
        case "Open Llama 7b":
            return "openlm-research/open_llama_7b"
        case "Open Llama 13b":
            return "openlm-research/open_llama_13b"
        case "BART Large CNN":
            return "facebook/bart-large-cnn"
        case "T5 XL NQ":
            return "google/t5-xl-ssm-nq"
        case "Flan T5 XL":
            return "google/flan-t5-xl"
        case _:
            raise Exception(f"Unknown model: {model}")


def parse_ensemble(
    path: str, device: str = "cpu"
) -> Tuple[WhiteboxModel, WhiteboxModel]:
    if os.path.exists(path):
        path = Path(path)
        model_paths = [model_dir for model_dir in path.iterdir()]
    else:
        model_paths = path.split(",")

    model = WhiteboxModel.from_pretrained(model_paths[0])
    ensemble_model = create_ensemble(model_paths=model_paths)

    return model, ensemble_model
