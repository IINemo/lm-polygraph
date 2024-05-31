from lm_polygraph.utils.manager import UEManager

datasets = ['coqa', 'triviaqa', 'xsum', 'mmlu', 'gsm8k', 'wmt14', 'wmt19']

for dataset in datasets:
    base_man = UEManager.load(f'polygraph_tacl_stablelm12b_{dataset}_train.man')
    cpmi_man = UEManager.load(f'polygraph_tacl_stablelm-2-12b_{dataset}_train_cpmi.man')
    
    key = ('sequence', 'MeanConditionalPointwiseMutualInformation')
    assert(key in base_man.estimations)
    assert(key in cpmi_man.estimations)

    base_man.estimations[key] = cpmi_man.estimations[key]
    base_man.save(f'polygraph_tacl_stablelm12b_{dataset}_train_updated.man')

