import subprocess


def bash(cmd):
    return subprocess.run(cmd, shell=True)


to_process = "coqa_verb_1s_topk,coqa_verb_2s_topk,mmlu_verb_1s_topk,mmlu_verb_2s_topk,triviaqa_verb_1s_topk,triviaqa_verb_2s_topk,wiki_bio,wmt14_deen,wmt14_fren,wmt19_deen"

for dataset in "wiki_bio".split(","):
    print("Processing dataset", dataset)
    bash(f"python3 manager.py --dataset {dataset} -p")
