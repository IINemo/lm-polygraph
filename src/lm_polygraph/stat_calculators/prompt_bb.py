import re
import torch
import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel

prompt_ue = f"""You will be given one summary written for a news article.

Your task is to estimate how you are uncertain in the generated summary.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Uncertainty (0-100) - the collective uncertainty of the generated text. This uncertainty should correlate with the overall quality of the generated text. The most uncertain instances should be erroneous, while the least uncertainty should be correct.

Evaluation Steps:

1. Read the news article carefully and identify the main topic and key points.
2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.
3. Assign a score for your uncertainty on a scale of 0 to 100, where 0 is the lowest and 100 is the highest based on the Evaluation Criteria.

Example:

Source Text: {{Document}}

Summary: {{Summary}}

Evaluation Form (scores ONLY):

- Uncertainty:"""

prompt_quality = f"""You will be given one summary written for a news article.
Your task is to estimate the overall quality of the generated summary.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Quality (0-100) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic". A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts. The summary should include only important information from the source document. Annotators were instructed to penalize summaries that contained redundancies and excess information.

Evaluation Steps:
1. Read the news article carefully and identify the main topic and key points.
2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order. The quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
3. Assign a score for quality on a scale of 0 to 100, where 0 is the lowest and 100 is the highest based on the Evaluation Criteria.

Example:
Source Text: The following reports have been waiting for your approval for more than 4 days.\nPlease review.\nOwner: Justin K Rostant Report Name: JRostant 10/24/01 Days In Mgr.Queue
Summary: AIG Exposure\n\nDate:\n\n12/12/2000\n\nTo:\n\nTanya Rohauer/HOU/ECT@ECT
Evaluation Form (scores ONLY): 
- Quality: 22
Explaining: Since the metric of the generated text is 0.22 by Rouge-L, the Quality is 22. 

Source Text: John:  The Enron deal at Elba Island amounts to a sendoout capacity of 160mmcf/d for 17 years.\nThis will be tough to fill as the majority of existing supply exceeds the max heat rate-I would look for Enron to get out of this in some way.\nEnron is currently having discussions with El Paso as to whether this contract goes into effect Oct 1st 2001 or Jan 1st 2002.\nElba Island has recently said that they will be running in Q4 although many still doubt it.\nAttached is a file that lists the terminals, their capacity owners, facility owners, expansion plans, and supply situation.\nLet me know if you have any more questions.
Summary: John:\n\nThe Enron deal at Elba Island amounts to a sendoout capacity of 160mmcf/d for 17 years.
Evaluation Form (scores ONLY): 
- Quality: 0
Explaining: Since the metric of the generated text is 0.0 by Rouge-L, the Quality is 0. Because the model does not summarize the text, and just rephrases it.

Source Text: {{Document}}
Summary: {{Summary}}
Evaluation Form (scores ONLY):
- Quality:"""


class PromptCalculatorBB(StatCalculator):
    """
    Calculates the probability for a specific token to be generated from the specific prompt.
    Used for P(True)-based methods.
    """

    def __init__(self, prompt: str, method: str):
        """
        Parameters:
            prompt (str): Prompt to use for estimating the answer of.
                The following values can be used in the prompt:
                - q: input text
                - a: generation text
                - s: list of several generation samples.
                Prompt example: 'Question: {q}. Is the following answer true? {a}'.
            expected (str): string to measure probability of. Must be decoded into one token,
                otherwise an exception will be raised.
            method (str): the name of the statistics to calculate with this calculator.
        """
        super().__init__([method], ["greedy_texts"])
        self.method = method
        self.prompt = prompt

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the probability for `expected` to be generated from `prompt`.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, consisting of:
                - 'greedy_texts' (List[str]): model generations for this batch,
                - 'sample_texts' (List[List[str]]): several sampling generations for each input text.
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - `method` (List[float]): logarithms of probability of generating `expected` from prompt formatted
                    at each input text.
        """
        answers = dependencies["greedy_texts"]
        inp_texts = [
            self.prompt.format(Document=text, Summary=ans)
            for text, ans in zip(texts, answers)
        ]

        batch: Dict[str, torch.Tensor] = model.tokenize(inp_texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}

        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                min_new_tokens=1,
                max_new_tokens=3,
                num_beams=1,
            )

        logits = torch.stack(out.scores, dim=1)
        num_scores = []
        for tokens in logits.argmax(-1).cpu().numpy():
            try:
                res_ue = model.tokenizer.decode(tokens)
                print(res_ue)
                res_ue_num = re.sub('[^\d\.]', '', res_ue)
                num_scores.append(int(res_ue_num))
            except:
                num_scores.append(100)
            
        return {self.method: np.array(num_scores)}
