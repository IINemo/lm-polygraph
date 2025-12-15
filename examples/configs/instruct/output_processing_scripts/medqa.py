import re


def normalize_medqa(s: str) -> str:
    """
    Extract the answer letter (A, B, C, or D) from the model output.
    Handles various formats like "A", "The answer is A", "A.", "Answer: A", etc.
    """
    if not s:
        return ""
    
    # Remove leading/trailing whitespace
    s = s.strip()
    
    # Try to find a single letter A, B, C, or D (case insensitive)
    # Look for the pattern at the start, after "Answer:", "answer:", or standalone
    patterns = [
        r'^[Aa]nswer\s*:?\s*([ABCD])',  # "Answer: A" or "answer: A"
        r'^([ABCD])[\.\)]?\s*$',  # "A." or "A)" or just "A"
        r'^[Tt]he\s+[Aa]nswer\s+is\s+([ABCD])',  # "The answer is A"
        r'\b([ABCD])\b',  # Any occurrence of A, B, C, or D
    ]
    
    for pattern in patterns:
        match = re.search(pattern, s, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in ['A', 'B', 'C', 'D']:
                return letter
    
    # If no pattern matches, try to extract the first A, B, C, or D found
    match = re.search(r'([ABCD])', s.upper())
    if match:
        return match.group(1)
    
    # Return empty string if no valid answer found
    return ""


def process_output_top1_medqa(output: str) -> str:
    """Process top-1 output for MedQA - extract just the answer letter."""
    return normalize_medqa(output)


def process_output_topk_medqa(output: str) -> str:
    """Process top-k output for MedQA - extract just the answer letter."""
    return normalize_medqa(output)


def process_output_cot_medqa(output: str) -> str:
    """Process chain-of-thought output for MedQA - extract just the answer letter."""
    return normalize_medqa(output)









