from dataclasses import dataclass

@dataclass
class GenerationParameters:
    temperature: float = 1.0
    topk: int = 1
    topp: float = 1.0
    do_sample: bool = False
    num_beams: int = 1
    repetition_penalty: float = 1
    allow_newlines: bool = True