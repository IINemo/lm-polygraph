from lm_polygraph.ue_metrics import *
from lm_polygraph.generation_metrics import *
from lm_polygraph.utils.manager import UEManager, _delete_nans
from lm_polygraph.ue_metrics.ue_metric import (
    UEMetric,
    get_random_scores,
    normalize_metric,
)
from pathlib import Path
import numpy as np
import time

from coqa import process_output_top1, process_output_topk, process_output_cot, process_target
from triviaqa import (process_output_top1 as process_output_top1_triviaqa, process_output_topk as process_output_topk_triviaqa, process_output_cot as process_output_cot_triviaqa, process_target as process_target_triviaqa)


def calculate_metrics(man):
    for (e_level, e_name), estimator_values in man.estimations.items():
        for (gen_level, gen_name), generation_metric in man.gen_metrics.items():
            del man.metrics[(e_level, e_name, gen_name, 'rpp')]
            del man.metrics[(e_level, e_name, gen_name, 'rpp_normalized')]
            del man.metrics[(e_level, e_name, gen_name, 'rcc-auc')]
            del man.metrics[(e_level, e_name, gen_name, 'rcc-auc_normalized')]
            for ue_metric in man.ue_metrics:
                if gen_level != e_level:
                    continue
                if len(estimator_values) != len(generation_metric):
                    raise Exception(
                        f"Got different number of metrics for {e_name} and {gen_name}: "
                        f"{len(estimator_values)} and {len(generation_metric)}"
                    )
                # TODO: Report how many nans!
                # This is important to know for a user
                ue, metric = _delete_nans(estimator_values, generation_metric)
                if len(ue) == 0:
                    man.metrics[e_level, e_name, gen_name, str(ue_metric)] = np.nan
                else:
                    oracle_score = ue_metric(-metric, metric)
                    bef = time.time()
                    random_score = get_random_scores(ue_metric, metric)
                    af = time.time()
                    print(ue_metric, ': ', af - bef)
                    ue_metric_val = ue_metric(ue, metric)
                    man.metrics[e_level, e_name, gen_name, str(ue_metric)] = (
                        ue_metric_val
                    )
                    man.metrics[
                        e_level, e_name, gen_name, str(ue_metric) + "_normalized"
                    ] = normalize_metric(ue_metric_val, oracle_score, random_score)


aligns = AlignScore(target_is_claims=True)

def new_ems_trivia(man):
    outs = man.stats['greedy_texts']
    targets = man.stats['target_texts']

    old_ems = man.gen_metrics[('sequence', 'Accuracy')]
    old_align = man.gen_metrics[('sequence', 'AlignScore')]
    ems = []
    align = []
    for out, target in zip(outs, targets):
        if prompt_type == 'top1':
            out = process_output_top1_triviaqa(out)
        elif prompt_type == 'topk':
            out = process_output_topk_triviaqa(out)
        elif prompt_type == 'cot':
            out = process_output_cot_triviaqa(out)
        local_ems = []
        local_align = []
        for t in target:
            t = process_target_triviaqa(t)
            local_ems.append(out == t)
            stats = {'greedy_texts': [out]}
            #local_align.append(aligns(stats, [t]))
        ems.append(any(local_ems))
        #align.append(max(local_align))

    ems = np.array(ems).astype(int)
    align = np.array(align)

    return old_ems, ems, old_align, align


def new_ems_coqa(man):
    outs = man.stats['greedy_texts']
    targets = man.stats['target_texts']

    old_ems = man.gen_metrics[('sequence', 'Accuracy')]
    old_align = man.gen_metrics[('sequence', 'AlignScore')]
    ems = []
    align = []
    for out, target in zip(outs, targets):
        target = process_target(target)
        if prompt_type == 'top1':
            out = process_output_top1(out)
        elif prompt_type == 'topk':
            out = process_output_topk(out)
        elif prompt_type == 'cot':
            out = process_output_cot(out)
        ems.append(out == target)
        stats = {'greedy_texts': [out]}
        #align.append(aligns(stats, [target]))

    ems = np.array(ems).astype(int)
#    align = np.array(align)

    return old_ems, ems, old_align, align


base_path = '/workspace/mans'

models = ['stable', 'mistral', 'gpt-4o-mini']
datasets = ['coqa', 'triviaqa']
prompt_types = ['top1', 'topk', 'cot']
ue_types = ['1s', '2s']

for model in models:
    for dataset in datasets:
        for prompt_type in prompt_types:
            for ue_type in ue_types:
                if prompt_type == 'cot' and ue_type == '1s':
                    continue
                man_name = f'{model}_{dataset}_verb_{ue_type}_{prompt_type}.man'
                man_path = Path(base_path) / man_name
                man = UEManager.load(man_path)

                if dataset == 'coqa':
                    old_ems, ems, old_align, align = new_ems_coqa(man)
                elif dataset == 'triviaqa':
                    old_ems, ems, old_align, align = new_ems_trivia(man)

                print('-' * 50)
                print(f'Model: {model}, Dataset: {dataset}, Prompt Type: {prompt_type}, UE Type: {ue_type}')
                print(f'Old Accuracy: {np.mean(old_ems)}')
                print(f'New Accuracy: {np.mean(ems)}')
                print(f'Old Align: {np.mean(old_align)}')
                print(f'New Align: {np.mean(align)}')

                man.gen_metrics[('sequence', 'Accuracy')] = ems
                man.ue_metrics = [
                    #ReversedPairsProportion(),
                    PredictionRejectionArea(),
                    PredictionRejectionArea(max_rejection=0.5),
                    #RiskCoverageCurveAUC(),
                ]
                calculate_metrics(man)
                man.save(Path(base_path) / f'{model}_{dataset}_verb_{ue_type}_{prompt_type}_official_em.man')


        man_name = f'{model}_{dataset}_ling_1s.man'
        man_path = Path(base_path) / man_name
        man = UEManager.load(man_path)

        if dataset == 'coqa':
            old_ems, ems, old_align, align = new_ems_coqa(man)
        elif dataset == 'triviaqa':
            old_ems, ems, old_align, align = new_ems_trivia(man)

        print('-' * 50)
        print(f'Model: {model}, Dataset: {dataset}, Prompt Type: Ling 1S')
        print(f'Old Accuracy: {np.mean(old_ems)}')
        print(f'New Accuracy: {np.mean(ems)}')
        print(f'Old Align: {np.mean(old_align)}')
        print(f'New Align: {np.mean(align)}')

        man.gen_metrics[('sequence', 'Accuracy')] = ems
        man.ue_metrics = [
            #ReversedPairsProportion(),
            PredictionRejectionArea(),
            PredictionRejectionArea(max_rejection=0.5),
            #RiskCoverageCurveAUC(),
        ]
        calculate_metrics(man)
        man.save(Path(base_path) / f'{model}_{dataset}_ling_1s_official_em.man')

        man_name = f'{model}_{dataset}_empirical_baselines.man'
        man_path = Path(base_path) / man_name
        man = UEManager.load(man_path)

        if dataset == 'coqa':
            old_ems, ems, old_align, align = new_ems_coqa(man)
        elif dataset == 'triviaqa':
            old_ems, ems, old_align, align = new_ems_trivia(man)

        print('-' * 50)
        print(f'Model: {model}, Dataset: {dataset}, Prompt Type: Baseline')
        print(f'Old Accuracy: {np.mean(old_ems)}')
        print(f'New Accuracy: {np.mean(ems)}')
        print(f'Old Align: {np.mean(old_align)}')
        print(f'New Align: {np.mean(align)}')

        man.gen_metrics[('sequence', 'Accuracy')] = ems
        man.ue_metrics = [
            #ReversedPairsProportion(),
            PredictionRejectionArea(),
            PredictionRejectionArea(max_rejection=0.5),
            #RiskCoverageCurveAUC(),
        ]
        calculate_metrics(man)
        man.save(Path(base_path) / f'{model}_{dataset}_empirical_baselines_official_em.man')
