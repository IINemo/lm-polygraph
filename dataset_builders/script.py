import subprocess


def bash(cmd):
    return subprocess.run(cmd, shell=True)


to_process = "person_bio_ar,person_bio_en,person_bio_ru,person_bio_zh"

for dataset in "xsum,coqa_ling_1s,coqa_verb_1s_top1,coqa_verb_1s_topk,coqa_verb_2s_cot,coqa_verb_2s_top1,coqa_verb_2s_topk,mmlu_ling_1s,mmlu_verb_1s_top1,mmlu_verb_1s_topk,mmlu_verb_2s_cot,mmlu_verb_2s_top1,mmlu_verb_2s_topk,triviaqa_ling_1s,triviaqa_verb_1s_top1,triviaqa_verb_1s_topk,triviaqa_verb_2s_cot,triviaqa_verb_2s_top1,triviaqa_verb_2s_topk,wiki_bio,wmt14_deen,wmt14_fren,wmt19_deen".split(","):
    print("Processing dataset", dataset)
    bash(f"python3 manager.py --dataset {dataset} -p")
