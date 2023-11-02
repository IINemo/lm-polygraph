from dataclasses import dataclass


@dataclass
class GenerationParameters:
    """
    Parameters to override in model generation.

    Parameters:
        temperature (float): Temperature in sampling generation. Has no effect when `do_sample` is not set.
            Default: 1.0.
        topk (int): Top-k token predictions to consider in sampling generation. Has no effect when `do_sample` is
            not set. Default: 1.
        topp (float): Only consider the highest unique tokens, which probabilities sum up to `topp`. Has no effect
            when `do_sample` is not set. Default: 1.0.
        do_sample (bool): If true, perform sampling from models probabilities. If false, only generate token with
            maximum probability. Default: False.
        num_beams (int): Number of beams if beam search generation is used. Has no effect when `do_sample` is not
            set. Default: 1.
        presence_penalty (float): Number between -2.0 and 2.0. Positive values penalize new tokens based on whether
            they appear in the text so far, increasing the model's likelihood to talk about new topics. Applied for
            OpenAI-API blackbox models. Default: 0.0.
        repetition_penalty (float): The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no
            penalty. Applied for whitebox models from HuggingFace. Default: 1.0.
        allow_newlines (bool): If set, the model is not allowed to generate tokens with newlines. Default: False.
    """

    temperature: float = 1.0
    topk: int = 1
    topp: float = 1.0
    do_sample: bool = False
    num_beams: int = 1
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    allow_newlines: bool = True
