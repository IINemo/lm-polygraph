import torch
import transformers
import argparse

from utils.manager import UEManager
from utils.dataset import Dataset
from utils.model import Model
from utils.processor import Logger
from generation_metrics import RougeMetric, WERTokenwiseMetric, BartScoreTokenwiseMetric, BartScoreSeqMetric
from estimators import *
from ue_metrics import ReversedPairsProportion, PredictionRejectionArea, RiskCoverageCurveAUC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default='/Users/ekaterinafadeeva/work/data/uncertainty_datasets/triviaqa.csv')
    parser.add_argument("--model", type=str, default='databricks/dolly-v2-3b')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", nargs='+', type=int)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_path", type=str,
                        default='/Users/ekaterinafadeeva/work/data/uncertainty_mans/debug')
    parser.add_argument("--cache_path", type=str,
                        default='/home/esfadeeva/data')
    args = parser.parse_args()

    if args.seed is None or len(args.seed) == 0:
        args.seed = [1]

    device = args.device
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for seed in args.seed:
        print('=' * 150)
        print('SEED:', seed)
        transformers.set_seed(seed)
        model = Model.from_pretrained(
            args.model,
            device=device,
        )
        dataset = Dataset.from_csv(
            args.dataset,
            'question', 'answer',
            batch_size=args.batch_size,
        )
        man = UEManager(
            dataset,
            model,
            [
                MaxProbabilitySeq(),
                MaxProbabilityNormalizedSeq(),
                EntropySeq(),
                MutualInformationSeq(),
                ConditionalMutualInformationSeq(tau=0.0656, lambd=3.599),
                AttentionEntropySeq(),
                AttentionRecursiveSeq(),
                ExponentialAttentionEntropySeq(coef=0.9),
                ExponentialAttentionEntropySeq(coef=0.8),
                ExponentialAttentionRecursiveSeq(coef=0.9),
                ExponentialAttentionRecursiveSeq(coef=0.8),
                PTrue(),
                PUncertainty(),
                PredictiveEntropy(),
                LengthNormalizedPredictiveEntropy(),
                LexicalSimilarity(metric='rouge1'),
                LexicalSimilarity(metric='rouge2'),
                LexicalSimilarity(metric='rougeL'),
                LexicalSimilarity(metric='BLEU'),
                SemanticEntropy(),
                PredictiveEntropyAdaptedSampling(),
                SemanticEntropyAdaptedSampling(),

                MaxProbabilityToken(),
                MaxProbabilityNormalizedToken(),
                EntropyToken(),
                MutualInformationToken(),
                ConditionalMutualInformationToken(tau=0.0656, lambd=3.599),
                AttentionEntropyToken(),
                AttentionRecursiveToken(),
                ExponentialAttentionEntropyToken(coef=0.9),
                ExponentialAttentionEntropyToken(coef=0.8),
                ExponentialAttentionRecursiveToken(coef=0.9),
                ExponentialAttentionRecursiveToken(coef=0.8),
                SemanticEntropyToken(model.model_path, args.cache_path),
            ],
            [
                RougeMetric('rouge1'),
                RougeMetric('rouge2'),
                RougeMetric('rougeL'),
                BartScoreSeqMetric('rh'),
                WERTokenwiseMetric(),
                BartScoreTokenwiseMetric('rh'),
            ],
            [
                ReversedPairsProportion(),
                PredictionRejectionArea(),
                RiskCoverageCurveAUC(),
            ],
            [
                Logger(),
            ],
        )
        man()
        man.save(args.save_path + f'_seed{seed}')
