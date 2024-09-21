import re
import string

from default import (TOP1_OUTPUT_IGNORE_REGEX,
                     TOPK_OUTPUT_IGNORE_REGEX,
                     CoT_OUTPUT_IGNORE_REGEX)


def normalize_em_triviaqa(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def process_target(target: str) -> str:
    target = normalize_em_triviaqa(target)
    return target


def process_output_top1(output: str) -> str:
    output = TOP1_OUTPUT_IGNORE_REGEX.sub("", output)
    output = normalize_em_triviaqa(output)
    return output


def process_output_topk(output: str) -> str:
    output = TOPK_OUTPUT_IGNORE_REGEX.sub("", output)
    output = normalize_em_triviaqa(output)
    return output


def process_output_cot(output: str) -> str:
    output = CoT_OUTPUT_IGNORE_REGEX.sub("", output)
    output = normalize_em_triviaqa(output)
    return output
