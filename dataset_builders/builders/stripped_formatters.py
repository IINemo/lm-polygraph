from typing import Sequence


def qa_stripped(question: str) -> str:
    q = (question or "").strip()
    return f"Q: {q}\nA:"


def mcqa_stripped(question: str, choices: Sequence[str]) -> str:
    q = (question or "").strip()
    labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
    lines = [f"Q: {q}"]
    for idx, choice in enumerate(choices):
        lab = labels[idx] if idx < len(labels) else str(idx + 1)
        lines.append(f"{lab}. {choice}")
    lines.append("A:")
    return "\n".join(lines)


def translation_stripped(original_text: str) -> str:
    t = (original_text or "").strip()
    return f"Original:\n{t}\nTranslation:"


def summarization_stripped(original_text: str) -> str:
    t = (original_text or "").strip()
    return f"Original:\n{t}\nSummary:"


def continuation_stripped(original_text: str) -> str:
    t = (original_text or "").strip()
    return f"Original:\n{t}\nContinuation:"
