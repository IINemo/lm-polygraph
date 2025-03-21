import argparse
import pathlib
import os
import matplotlib.pyplot as plt
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.estimators import *
from lm_polygraph.ue_metrics import *
from lm_polygraph.stat_calculators import *
from tqdm import tqdm
import pickle
import numpy as np

estimators = [
    GreedySemanticEnrichedPPLAveDissimilarity(),
    SupervisedGreedySemanticEnrichedPPLAveDissimilarity(),
    GreedySemanticEnrichedMaxprobAveDissimilarity(),
    SupervisedGreedySemanticEnrichedMaxprobAveDissimilarity(),
    GreedySemanticEnrichedMTEAveDissimilarity(),
    SupervisedGreedySemanticEnrichedMTEAveDissimilarity(),
]
names = {
    "GreedySemanticEnrichedPPLAveDissimilarity": 'PPL Cocoa',
    "SupervisedGreedySemanticEnrichedPPLAveDissimilarity": 'Supervised PPL Cocoa',
    "GreedySemanticEnrichedMaxprobAveDissimilarity": 'Maxprob Cocoa',
    "SupervisedGreedySemanticEnrichedMaxprobAveDissimilarity": 'Supervised Maxprob Cocoa',
    "GreedySemanticEnrichedMTEAveDissimilarity":  'MTE Cocoa',
    "SupervisedGreedySemanticEnrichedMTEAveDissimilarity": 'Supervised MTE Cocoa',
}
ue_metrics = [
    #PredictionRejectionArea(),
    PredictionRejectionArea(max_rejection=0.5),
]
gen_metric = 'AlignScoreOutputTarget'

models = ["llama8b"]
datasets = ["coqa"]

preds = pickle.load(open("test_predictions.pickle", "rb"))['predictions']

script_dir = '.'
out_dir = '.'

# Loop through each model and dataset combination
for model in models:
    for dataset in datasets:
        # Construct manager file path
        manager_filename = f"{model}_{dataset}.man"
        manager_path = os.path.join(script_dir, manager_filename)

        man = UEManager.load(manager_path)

        stats = man.stats
        stats["greedy_sentence_similarity_pred"] = preds

        man.estimations = {}
        for estimator in estimators:
            values = estimator(stats)
            man.estimations[('sequence', str(estimator))] = values

        man.stats = stats

        man.ue_metrics = ue_metrics

        man.eval_ue()
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        man.save_path = os.path.join(out_dir, f"{model}_{dataset}_supervised.man")
        man.save()
        
        base_metric = np.mean(man.gen_metrics[('sequence', gen_metric)])
        print(f'Base metric ({gen_metric}): {base_metric:.3f}')
        print('='*50)
        for i, est in enumerate(estimators):
            if i % 2 == 0 and i != 0:
                print('-'*50)
            prr = man.metrics[('sequence', str(est), gen_metric, 'prr_0.5_normalized')]
            print(f"{names[str(est)]}: {prr:.3f}")
