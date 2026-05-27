import re


# Noise stripped before any answer extraction.
GEMMA_OUTPUT_IGNORE_REGEX = re.compile(r"<end_of_turn>")
QWEN_FALCON_EOS_IGNORE_REGEX = re.compile(r"<\|im_end\|>|<\|endoftext\|>")
# Falcon chat templates use plain-text turn markers; if generation did not
# stop at our generate_until strings, drop everything from the next user turn
# onward. The colon is optional because some outputs end with 'User ' alone.
FALCON_TURN_LEAK_REGEX = re.compile(r"\n\n?User\b.*$", re.DOTALL)
# Strip everything up to and including '### Answer:' so reasoning-style
# prompts keep only the answer portion. Whitespace between '###' and 'Answer'
# is permissive (covers '###\nAnswer:').
REASONING_OUTPUT_IGNORE_REGEX = re.compile(r"(?s).*###\s*Answer:\s*")
# Leading '###' prefix without an 'Answer:' word (e.g. '### a)'). Applied
# only after REASONING_OUTPUT_IGNORE_REGEX has had its chance.
HASH_PREFIX_REGEX = re.compile(r"^\s*#{2,}\s*")
# Common preambles models like to emit before the answer.
LEADING_PHRASE_REGEX = re.compile(
    r"^\s*(?:the\s+answer\s+is|final\s+answer|answer)\s*[:\-]?\s*",
    re.IGNORECASE,
)

# A single MCQ letter (a-d), optionally wrapped in parens/brackets and
# optionally followed by ) ] . : , or whitespace, anchored at the start of
# the cleaned text.
MCQ_LETTER_REGEX = re.compile(
    r"^\s*[\(\[]?\s*([a-dA-D])\s*[\)\]\.\:\,]?(?:\s|$)"
)
# Fallback: search anywhere for 'the answer is X' / 'answer: X' / '(X)'.
ANSWER_PHRASE_SEARCH_REGEX = re.compile(
    r"(?i)(?:the\s+answer\s+is|answer\s*[:\-])\s*[\(\[]?\s*([a-dA-D])\b"
)
PAREN_LETTER_SEARCH_REGEX = re.compile(r"\(([a-dA-D])\)")

# Integer with optional sign and thousands separators; decimals truncated.
INTEGER_EXTRACTION_REGEX = re.compile(r"-?\d[\d,]*")

# Legacy regex kept for back-compat with old configs.
PARENTHESEIS_OUTPUT_IGNORE_REGEX = re.compile(r"\)")


def _strip_noise(output: str) -> str:
    output = GEMMA_OUTPUT_IGNORE_REGEX.sub("", output)
    output = QWEN_FALCON_EOS_IGNORE_REGEX.sub("", output)
    output = FALCON_TURN_LEAK_REGEX.sub("", output)
    output = REASONING_OUTPUT_IGNORE_REGEX.sub("", output)
    output = HASH_PREFIX_REGEX.sub("", output)
    output = LEADING_PHRASE_REGEX.sub("", output)
    return output.strip()


def process_output_mcq(output: str) -> str:
    """Extract a single multiple-choice letter (a-d) for MMLU/medmcqa direct.

    Handles 'a', 'a)', '(a)', 'A', 'Answer: a', 'The answer is c.',
    'a) Coronary vasodilation', '### a)', '### a) Paap', and Falcon
    turn-marker leaks like 'a\\nUser'.
    """
    cleaned = _strip_noise(output)
    m = MCQ_LETTER_REGEX.match(cleaned)
    if m:
        return m.group(1).lower()
    m = ANSWER_PHRASE_SEARCH_REGEX.search(cleaned)
    if m:
        return m.group(1).lower()
    m = PAREN_LETTER_SEARCH_REGEX.search(cleaned)
    if m:
        return m.group(1).lower()
    return cleaned


def process_output_number(output: str) -> str:
    """Extract a first integer for gsm8k direct.

    Normalizes thousands separators ('1,234' -> '1234'). Does not attempt
    letter extraction so 'a 5' style outputs still resolve to '5'.
    """
    cleaned = _strip_noise(output)
    m = INTEGER_EXTRACTION_REGEX.search(cleaned)
    if m:
        return m.group(0).replace(",", "")
    return cleaned


def process_output(output: str) -> str:
    """General-purpose processor. Tries number first (legacy behavior),
    then MCQ letter, then falls back to cleaned text.

    Prefer process_output_mcq or process_output_number when the task type
    is known; this function exists for back-compat with configs that don't
    pin fn_name."""
    cleaned = _strip_noise(output)
    cleaned_legacy = PARENTHESEIS_OUTPUT_IGNORE_REGEX.sub("", cleaned)

    m = INTEGER_EXTRACTION_REGEX.search(cleaned_legacy)
    if m:
        return m.group(0).replace(",", "")

    m = MCQ_LETTER_REGEX.match(cleaned)
    if m:
        return m.group(1).lower()
    m = ANSWER_PHRASE_SEARCH_REGEX.search(cleaned)
    if m:
        return m.group(1).lower()

    return cleaned_legacy


def process_target(output: str) -> str:
    """Target passthrough. Lowercase + strip so case/whitespace differences
    in dataset labels don't fight the processors above."""
    return str(output).strip().lower()
