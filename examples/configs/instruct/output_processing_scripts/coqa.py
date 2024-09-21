import re
import string

from default import (TOP1_OUTPUT_IGNORE_REGEX,
                     TOPK_OUTPUT_IGNORE_REGEX,
                     CoT_OUTPUT_IGNORE_REGEX)


def normalize_em_coqa(s: str) -> str:
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def process_target(target: str) -> str:
    target = normalize_em_coqa(target)
    return target


def process_output_top1(output: str) -> str:
    output = TOP1_OUTPUT_IGNORE_REGEX.sub("", output)
    output = normalize_em_coqa(output)
    return output


def process_output_topk(output: str) -> str:
    output = TOPK_OUTPUT_IGNORE_REGEX.sub("", output)
    output = normalize_em_coqa(output)
    return output


def process_output_cot(output: str) -> str:
    output = CoT_OUTPUT_IGNORE_REGEX.sub("", output)
    output = normalize_em_coqa(output)
    return output
