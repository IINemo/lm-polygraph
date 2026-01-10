#!/usr/bin/env python
import argparse
import sys
from typing import Dict, Tuple, List

import datasets


# Mapping from dataset key to (dataset_repo_name, subset)
DATASET_MAP: Dict[str, Tuple[str, str]] = {
    # base
    "trivia_qa_tiny": ("trivia_qa_tiny", "continuation"),
    "aeslc": ("aeslc", "continuation"),
    # babi_qa
    "babi_qa": ("babi_qa", "continuation"),
    # coqa
    "coqa": ("coqa", "continuation"),
    "coqa_empirical_baselines": ("coqa", "empirical_baselines"),
    "coqa_ling_1s": ("coqa", "ling_1s"),
    "coqa_verb_1s_top1": ("coqa", "verb_1s_top1"),
    "coqa_verb_1s_topk": ("coqa", "verb_1s_topk"),
    "coqa_verb_2s_cot": ("coqa", "verb_2s_cot"),
    "coqa_verb_2s_top1": ("coqa", "verb_2s_top1"),
    "coqa_verb_2s_topk": ("coqa", "verb_2s_topk"),
    "coqa_simple_instruct": ("coqa", "simple_instruct"),
    # gsm8k
    "gsm8k": ("gsm8k", "continuation"),
    "gsm8k_simple_instruct": ("gsm8k", "simple_instruct"),
    # mmlu
    "mmlu": ("mmlu", "continuation"),
    "mmlu_empirical_baselines": ("mmlu", "empirical_baselines"),
    "mmlu_ling_1s": ("mmlu", "ling_1s"),
    "mmlu_verb_1s_top1": ("mmlu", "verb_1s_top1"),
    "mmlu_verb_1s_topk": ("mmlu", "verb_1s_topk"),
    "mmlu_verb_2s_cot": ("mmlu", "verb_2s_cot"),
    "mmlu_verb_2s_top1": ("mmlu", "verb_2s_top1"),
    "mmlu_verb_2s_topk": ("mmlu", "verb_2s_topk"),
    "mmlu_simple_instruct": ("mmlu", "simple_instruct"),
    # person
    "person_bio_ar": ("person_bio", "ar"),
    "person_bio_en": ("person_bio", "en"),
    "person_bio_ru": ("person_bio", "ru"),
    "person_bio_zh": ("person_bio", "zh"),
    # triviaqa
    "triviaqa": ("triviaqa", "continuation"),
    "triviaqa_empirical_baselines": ("triviaqa", "empirical_baselines"),
    "triviaqa_ling_1s": ("triviaqa", "ling_1s"),
    "triviaqa_verb_1s_top1": ("triviaqa", "verb_1s_top1"),
    "triviaqa_verb_1s_topk": ("triviaqa", "verb_1s_topk"),
    "triviaqa_verb_2s_cot": ("triviaqa", "verb_2s_cot"),
    "triviaqa_verb_2s_top1": ("triviaqa", "verb_2s_top1"),
    "triviaqa_verb_2s_topk": ("triviaqa", "verb_2s_topk"),
    "triviaqa_simple_instruct": ("triviaqa", "simple_instruct"),
    # wiki
    "wiki_bio": ("wiki_bio", "continuation"),
    # wmt
    "wmt14_deen": ("wmt14", "deen"),
    "wmt14_fren": ("wmt14", "fren"),
    "wmt14_fren_simple_instruct": ("wmt14", "fren_simple_instruct"),
    "wmt19_deen": ("wmt19", "deen"),
    "wmt19_deen_simple_instruct": ("wmt19", "deen_simple_instruct"),
    "wmt19_ruen": ("wmt19", "ruen"),
    "wmt19_ruen_simple_instruct": ("wmt19", "ruen_simple_instruct"),
    # truthfulqa
    "truthfulqa": ("truthfulqa", "continuation"),
    "truthfulqa_simple_instruct": ("truthfulqa", "simple_instruct"),
    # samsum
    "samsum": ("samsum", "continuation"),
    "samsum_simple_instruct": ("samsum", "simple_instruct"),
    # xsum
    "xsum": ("xsum", "continuation"),
    "xsum_simple_instruct": ("xsum", "simple_instruct"),
}


def compare_columns(
    a_ds: datasets.Dataset, b_ds: datasets.Dataset, col: str
) -> List[int]:
    diffs: List[int] = []
    a_col = a_ds[col]
    b_col = b_ds[col]
    if len(a_col) != len(b_col):
        return list(range(max(len(a_col), len(b_col))))
    for i, (x, y) in enumerate(zip(a_col, b_col)):
        if x != y:
            diffs.append(i)
            if len(diffs) >= 5:
                break
    return diffs


def main():
    p = argparse.ArgumentParser(
        description="Validate updated datasets against reference for input/output equality"
    )
    p.add_argument(
        "--ref-namespace",
        default="LM-Polygraph",
        help="Reference namespace (default: LM-Polygraph)",
    )
    p.add_argument(
        "--new-namespace",
        required=True,
        help="Namespace with updated datasets to validate",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of dataset keys to validate (defaults to all)",
    )
    args = p.parse_args()

    keys = args.datasets or list(DATASET_MAP.keys())
    any_errors = False

    for key in keys:
        if key not in DATASET_MAP:
            print(f"[SKIP] Unknown dataset key: {key}")
            continue
        dataset_name, subset = DATASET_MAP[key]
        ref_repo = f"{args.ref_namespace}/{dataset_name}"
        new_repo = f"{args.new_namespace}/{dataset_name}"

        print(f"=== Checking {key} (subset={subset}) ===")
        try:
            ref = datasets.load_dataset(ref_repo, subset)
            new = datasets.load_dataset(new_repo, subset)
        except Exception as e:
            print(f"[ERROR] Failed to load datasets for {key}: {e}")
            any_errors = True
            continue

        ref_splits = set(ref.keys())
        new_splits = set(new.keys())
        common = sorted(ref_splits & new_splits)
        if not common:
            print(
                f"[WARN] No common splits. ref={sorted(ref_splits)} new={sorted(new_splits)}"
            )
            continue

        for split in common:
            ref_split = ref[split]
            new_split = new[split]

            # Row count check first
            if len(ref_split) != len(new_split):
                print(
                    f"[ERR] {key}/{split}: row count mismatch: {len(ref_split)} != {len(new_split)}"
                )
                any_errors = True
                continue

            for col in ("input", "output"):
                if (
                    col not in ref_split.column_names
                    or col not in new_split.column_names
                ):
                    print(
                        f"[ERR] {key}/{split}: column '{col}' missing in one of the datasets"
                    )
                    any_errors = True
                    continue
                diffs = compare_columns(ref_split, new_split, col)
                if diffs:
                    print(
                        f"[ERR] {key}/{split}: column '{col}' differs at indices: {diffs[:5]} (showing up to 5)"
                    )
                    any_errors = True
                else:
                    print(f"[OK]  {key}/{split}: column '{col}' matches")

    if any_errors:
        print("\nValidation completed with errors.")
        sys.exit(1)
    else:
        print("\nAll checked datasets match for input/output columns.")


if __name__ == "__main__":
    main()
