import copy
import os
import torch
import transformers
import argparse

from utils.manager import UEManager
from utils.dataset import Dataset
from utils.model import Model
from utils.processor import Logger
from generation_metrics import RougeMetric, WERTokenwiseMetric, BartScoreSeqMetric, ModelScoreSeqMetric, ModelScoreTokenwiseMetric
from estimators import *
from ue_metrics import ReversedPairsProportion, PredictionRejectionArea, RiskCoverageCurveAUC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default='/Users/ekaterinafadeeva/work/data/uncertainty_datasets/triviaqa.csv')
    parser.add_argument("--train_dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default='databricks/dolly-v2-3b')
    parser.add_argument("--use_density_based_ue", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--density_based_params_path", type=str, default="/home/jovyan/projects/lm-polygraph/workdir")
    parser.add_argument("--subsample_train_dataset", type=int, default=10)
    parser.add_argument("--subsample_eval_dataset", type=int, default=-1)
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

        dataset = Dataset.load(
            args.dataset,
            'question', 'answer',
            batch_size=args.batch_size,
        )
                        
        if args.use_density_based_ue:
            if (args.train_dataset is not None) and (args.train_dataset != args.dataset):
                train_dataset = Dataset.load(
                    args.train_dataset,
                    'question', 'answer',
                    batch_size=args.batch_size,
                )
            else:
                X_train, X_test, y_train, y_test = dataset.train_test_split(
                    test_size=0.7,
                    seed=seed,
                    split="eval"
                )
                train_dataset = Dataset(x=X_train, y=y_train, batch_size=args.batch_size)
                
            if args.subsample_train_dataset != -1:
                train_dataset.subsample(args.subsample_train_dataset, seed=seed)
        
            if model.model_type == "Seq2SeqLM":
                density_based_ue = [
                    MahalanobisDistanceSeq("encoder", parameters_path=f"{args.density_based_params_path}/md_encoder/{args.model.split('/')[-1]}"),
                    MahalanobisDistanceSeq("decoder", parameters_path=f"{args.density_based_params_path}/md_decoder/{args.model.split('/')[-1]}"),
                    RDESeq("encoder", parameters_path=f"{args.density_based_params_path}/rde_encoder/{args.model.split('/')[-1]}"),
                    RDESeq("decoder", parameters_path=f"{args.density_based_params_path}/rde_decoder/{args.model.split('/')[-1]}"),
                ]
            else:
                density_based_ue = [
                    MahalanobisDistanceSeq("decoder", parameters_path=f"{args.density_based_params_path}/md_decoder/{args.model.split('/')[-1]}"),
                    RDESeq("decoder", parameters_path=f"{args.density_based_params_path}/rde_decoder/{args.model.split('/')[-1]}"),
                ]
        else:
            train_dataset = None
            density_based_ue = []
            
        if args.subsample_eval_dataset != -1:
            dataset.subsample(args.subsample_eval_dataset, seed=seed)
            
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
            ] + density_based_ue,
            [
                RougeMetric('rouge1'),
                RougeMetric('rouge2'),
                RougeMetric('rougeL'),
                BartScoreSeqMetric('rh'),
                ModelScoreSeqMetric('model_rh'),
                ModelScoreTokenwiseMetric('model_rh'),
                WERTokenwiseMetric(),
            ],
            [
                ReversedPairsProportion(),
                PredictionRejectionArea(),
                RiskCoverageCurveAUC(),
            ],
            [
                Logger(),
            ],
            train_data=train_dataset,
        )
        man()
        man.save(args.save_path + f'_seed{seed}')
