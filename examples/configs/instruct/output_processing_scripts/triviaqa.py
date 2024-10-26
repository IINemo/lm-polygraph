import re
import string

TOP1_OUTPUT_IGNORE_REGEX = re.compile(r"(?s)[Gg]uess:|[\n\.\(\,].*")
TOPK_OUTPUT_IGNORE_REGEX = re.compile(r"(?s)G1:|[\n\.\(\,].*")
CoT_OUTPUT_IGNORE_REGEX = re.compile(r"(?s).*[Gg]uess:|[\n\.\(\,].*")


def normalize_em_triviaqa(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(
        remove_articles(handle_punc(lower(replace_underscore(s))))
    ).strip()


def process_output_top1_triviaqa(output: str) -> str:
    output = TOP1_OUTPUT_IGNORE_REGEX.sub("", output)
    output = normalize_em_triviaqa(output)
    return output


def process_output_topk_triviaqa(output: str) -> str:
    output = TOPK_OUTPUT_IGNORE_REGEX.sub("", output)
    output = normalize_em_triviaqa(output)
    return output


def process_output_cot_triviaqa(output: str) -> str:
    output = CoT_OUTPUT_IGNORE_REGEX.sub("", output)
    output = normalize_em_triviaqa(output)
    return output
