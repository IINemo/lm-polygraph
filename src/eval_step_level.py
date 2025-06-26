import time
import torch
import os
import random
import numpy as np
import transformers
import wandb
from argparse import ArgumentParser, BooleanOptionalAction, ArgumentTypeError


from lm_polygraph.stat_calculators.step.steps_extractor import StepsExtractor
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.estimators import Estimator
from lm_polygraph.estimators.claim.claim_conditioned_probability import ClaimConditionedProbabilityClaim
from lm_polygraph.estimators.claim.frequency_scoring import FrequencyScoringClaim
from lm_polygraph.estimators.claim.p_true import PTrueClaim
from lm_polygraph.estimators.claim.perplexity import PerplexityClaim
from lm_polygraph.estimators.claim.random_baseline import RandomBaselineClaim
from lm_polygraph.estimators.claim.token_entropy import MaxTokenEntropyClaim
from lm_polygraph.stat_calculators import StatCalculator, GreedyProbsCalculator, EntropyCalculator, \
    ClaimPromptCalculator, GreedyAlternativesNLICalculator, GreedyAlternativesFactPrefNLICalculator, \
    SamplingGenerationCalculator
from lm_polygraph.estimators.claim.max_probability import MaximumClaimProbability
from lm_polygraph.stat_calculators.semantic_classes_claim_to_samples import SemanticClassesClaimToSamplesCalculator
from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.estimators.step.degmat import StepsDegMat
from lm_polygraph.estimators.step.eccentricity import StepsEccentricity
from lm_polygraph.stat_calculators.step.greedy_nli_similarity import StepsGreedyNLISimilarityCalculator
from lm_polygraph.estimators.step.lexical_similarity import StepsLexicalSimilarity
from lm_polygraph.estimators.step.num_sem_sets import StepsNumSemSets
from lm_polygraph.stat_calculators.step.semantic_classes import StepsSemanticClassesCalculator
from lm_polygraph.estimators.step.semantic_entropy import StepsSemanticEntropy
from lm_polygraph.estimators.step.dissimilarity import StepsDissimilarity
from lm_polygraph.stat_calculators.step.semantic_matrix import StepsSemanticMatrixCalculator
from lm_polygraph.stat_calculators.step.stepwise_sampling import StepwiseSamplingCalculator

EXCLUDE_SAVE_STATS: list[str] = [
    "embeddings",
    "embeddings_encoder",
    "embeddings_decoder",
    "attention_all",
    "tokenizer",
    "greedy_log_probs",
    "sample_sentence_similarity",
    "sample_token_similarity",
    "sample_embeddings",
]

def parse_tuple(s):
    try:
        parts = s.strip("()").split(",")
        return tuple(part.strip() for part in parts)
    except Exception:
        raise ArgumentTypeError("Tuple must be in the form: value1,value2")


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='HF path to base model')
    parser.add_argument('--dataset-path', type=parse_tuple, required=True, help='Path to HF dataset with questions')
    parser.add_argument('--dataset-split', type=str, required=True, help='HF dataset split')
    parser.add_argument('--prompt-path', type=str, required=False, help='Path to prompt')
    parser.add_argument('--device', type=str, default='auto', help='Torch device to run experiments on')
    parser.add_argument('--max-new-tokens', type=int, default=256, help='Max number of new generated tokens')
    parser.add_argument('--save-path', type=str, required=True, help='Path to save manager')
    parser.add_argument('--finetuned-deberta-path', type=str, default=None,
                        help='Path to fine-tuned version of deberta')
    parser.add_argument('--deberta-batch-size', type=int, default=10, help='Batch size for deberta')
    parser.add_argument('--wandb-project', type=str, default='tot-decoding', help='WandB project name')
    parser.add_argument('--n-samples', type=int, default=5, help='Number of sample chains and steps')
    parser.add_argument('--verbose', action=BooleanOptionalAction, default=False)
    return parser


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)


def main(args):
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
    )

    model = WhiteboxModel.from_pretrained(args.model_path, torch_dtype="auto", device_map=args.device)
    data = Dataset.from_datasets(
        args.dataset_path,
        x_column='question',
        y_column='answer',
        prompt=open(args.prompt_path).read() if args.prompt_path else "",
        split=args.dataset_split,
    )
    nli_model = Deberta(batch_size=args.deberta_batch_size)
    if args.finetuned_deberta_path:
        state_dict = torch.load(args.finetuned_deberta_path)
        nli_model._deberta.load_state_dict(state_dict)

    stat_calculators: list[StatCalculator] = [
        GreedyProbsCalculator(),
        StepsExtractor(),
        EntropyCalculator(),
        ClaimPromptCalculator(),
        SamplingGenerationCalculator(samples_n=args.n_samples),
        SemanticClassesClaimToSamplesCalculator(nli_model),
        GreedyAlternativesNLICalculator(nli_model),
        GreedyAlternativesFactPrefNLICalculator(nli_model),
        StepwiseSamplingCalculator(candidates_per_step=args.n_samples),
        StepsSemanticMatrixCalculator(nli_model),
        StepsSemanticClassesCalculator(),
        StepsGreedyNLISimilarityCalculator(nli_model),
    ]
    estimators: list[Estimator] = [
        RandomBaselineClaim(),
        MaximumClaimProbability(),
        MaxTokenEntropyClaim(),
        PerplexityClaim(),
        PTrueClaim(),
        ClaimConditionedProbabilityClaim(nli_context="no_context"),
        ClaimConditionedProbabilityClaim(nli_context="fact_pref"),
        FrequencyScoringClaim(),
        StepsSemanticEntropy(),
        StepsLexicalSimilarity(),
        StepsDegMat(),
        StepsEccentricity(),
        StepsNumSemSets(),
        StepsDissimilarity('rougeL'),
        StepsDissimilarity('nli_entail'),
        StepsDissimilarity('nli_contra'),
        StepsDissimilarity('nli_ccp'),
    ]
    man: dict = {
        'stats': [],
        'estimates': [],
    }
    if os.path.exists(args.save_path):
        man = torch.load(args.save_path)
    for i, (input_texts, target_texts) in enumerate(data):
        if len(man['estimates']) > i:
            print(f"Skipping batch#{i}")
            continue
        set_seed(228)
        if args.verbose:
            print(f'input_texts: {input_texts}')
        stats: dict = {
            "input_texts": input_texts,
            "target_texts": target_texts,
        }
        for stat_calculator in stat_calculators:
            name = stat_calculator.__class__.__name__
            print(f"Calculating {name}...")
            start_time = time.time()
            result = stat_calculator(stats, input_texts, model, max_new_tokens=args.max_new_tokens)
            elapsed = time.time() - start_time
            stats.update(result)
            print(f"Done calculating in {elapsed:.2f} seconds...")
            wandb.log({f"timing/stat_calculator/{name}": elapsed})
            if args.verbose:
                for key, val in result.items():
                    if key in EXCLUDE_SAVE_STATS:
                        continue
                    print(f"{key}: {val}")

        estimates: dict[str, list] = {}
        for estimator in estimators:
            name = str(estimator)
            print(f"Estimating {name}...")
            start_time = time.time()
            result = estimator(stats)
            elapsed = time.time() - start_time
            estimates[name] = result
            print(f"Done estimating in {elapsed:.2f} seconds...")
            wandb.log({f"timing/estimator/{name}": elapsed})
            if args.verbose:
                print(f"{estimator}: {result}")

        man['stats'].append({k: v for k, v in stats.items() if k not in EXCLUDE_SAVE_STATS})
        man['estimates'].append(estimates)

        print(f"Saving to {args.save_path}...")
        torch.save(man, args.save_path)

        wandb.log({f"iteration": i + 1})

        # artifact = wandb.Artifact('results', type='model_output')
        # artifact.add_file(args.save_path)
        # wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
