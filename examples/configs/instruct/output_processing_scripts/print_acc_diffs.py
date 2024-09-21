from lm_polygraph.utils.manager import UEManager
from pathlib import Path
import numpy as np

from coqa import normalize_em_coqa
from triviaqa import normalize_em_triviaqa

base_path = '/Users/romanvashurin/workspace/normalization_effects/mans'

models = ['stablelm12b', 'mistral7b']
datasets = ['coqa', 'triviaqa']

for model in models:
    for dataset in datasets:
        man_name = f'polygraph_tacl_{model}_{dataset}.man'
        man_path = Path(base_path) / man_name
        man = UEManager.load(man_path)

        outs = man.stats['greedy_texts']
        targets = man.stats['target_texts']
        
        old_ems = man.gen_metrics[('sequence', 'Accuracy')]
        ems = []
        for out, target in zip(outs, targets):
            if dataset == 'coqa':
                out = normalize_em_coqa(out)
                target = normalize_em_coqa(target)
                ems.append(out == target)
            elif dataset == 'triviaqa':
                out = normalize_em_triviaqa(out)
                local_ems = []
                for t in target:
                    t = normalize_em_triviaqa(t)
                    local_ems.append(out == t)
                ems.append(any(local_ems))
        
        print('-' * 50)
        print(f'Model: {model}, Dataset: {dataset}')
        print(f'Old Accuracy: {np.mean(old_ems)}')
        print(f'New Accuracy: {np.mean(ems)}')
