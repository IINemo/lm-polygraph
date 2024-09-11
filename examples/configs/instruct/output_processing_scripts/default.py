import re
import string

TOP1_OUTPUT_IGNORE_REGEX = re.compile(r"(?s)[Gg]uess:|[\n\.\(\,].*")
TOPK_OUTPUT_IGNORE_REGEX = re.compile(r"(?s)G1:|[\n\.\(\,].*")
CoT_OUTPUT_IGNORE_REGEX = re.compile(r"(?s).*[Gg]uess:|[\n\.\(\,].*")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def process_target(target: str) -> str:
    target = normalize_text(target)
    return target


def process_output_top1(output: str) -> str:
    output = TOP1_OUTPUT_IGNORE_REGEX.sub("", output)
    output = normalize_text(output)
    return output


def process_output_topk(output: str) -> str:
    output = TOPK_OUTPUT_IGNORE_REGEX.sub("", output)
    output = normalize_text(output)
    return output


def process_output_cot(output: str) -> str:
    output = CoT_OUTPUT_IGNORE_REGEX.sub("", output)
    output = normalize_text(output)
    return output
