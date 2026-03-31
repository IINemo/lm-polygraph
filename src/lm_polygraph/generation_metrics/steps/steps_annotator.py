import numpy as np

from lm_polygraph.generation_metrics.openai_fact_check import *
from lm_polygraph.stat_calculators.extract_claims import *


class StepsAnnotator(GenerationMetric):
    def __init__(
            self,
            cache_path: str = "~/.cache",
            model: str = 'gpt-4o',
            api_key: str | None = None,
            progress_bar: bool = True,
            n_threads: int = 1,
            wait_times: tuple = (5, 10, 30, 60, 120),
    ):
        super().__init__(["input_texts", "claims"], "claim")

        self.chat = OpenAIChat(openai_model=model, cache_path=cache_path, api_key=api_key, wait_times=wait_times)

        self.model = model
        self.progress_bar = progress_bar
        self.n_threads = n_threads

    def __str__(self):
        return f"StepsAnnotator {self.model}"

    def parse_problem(self, input_text: str):
        return input_text.split('<Question>: ', 1)[-1].split('<|im_end|>')[0]

    def prompt1(self, input_text: str, claims: list[Claim], answer: str) -> str:
        problem = self.parse_problem(input_text)
        steps = '\n'.join([cl.claim_text.strip() for i, cl in enumerate(claims)])
        return r'''You are given a problem, a ground-truth solution, and a step-by-step student solution. Your task is to analyze each step in the student’s solution to determine whether it is both logically correct and relevant.

Instructions:
- Carefully examine each student step for logical errors or unnecessary/redundant reasoning.
- If all steps are correct and they lead to the same final answer as the ground-truth solution, conclude that there are no errors.
- If any step contains an error that would prevent the student from reaching the correct solution, identify and report those specific steps with an explanation.

PROBLEM:
{problem}

GROUND-TRUTH SOLUTION:
{answer}

STUDENT'S SOLUTION STEPS:
{steps}

Now, please evaluate whether the student’s steps are correct and logical.'''.format(problem=problem, answer=answer,
                                                                                    steps=steps)

    def prompt2(self, input_text: str, claims: list[Claim], answer: str, reply: str) -> str:
        problem = self.parse_problem(input_text)
        steps = [cl.claim_text.strip() for i, cl in enumerate(claims)]
        return r"""
You are given:
- A problem
- A student's step-by-step solution (as a Python list of string steps)
- An assessment of student's solution

Your task:
Output a single Python list where each element is:
- 1 if the corresponding step is correct
- 0 if the step is incorrect

Important:
- Output only the list, nothing else.
- The list must have the same length as the number of steps.

PROBLEM:
{problem}

STUDENT'S SOLUTION STEPS:
{steps}

ASSESSMENT OF STUDENT SOLUTION STEPS:
{reply}

OUTPUT LIST:
""".format(problem=problem, steps=steps, reply=reply)

    def parse_reply(self, reply: str) -> list[int] | None:
        if 'all steps are correct' in reply.lower():
            return []
        orig_reply = reply
        reply = reply.strip().replace(' ', '').replace('Step', '')
        if '```python' in reply:
            reply = reply.split('```python')[-1].split('```')[0].strip()
        if reply.startswith('[') and reply.endswith(']'):
            reply = reply[1:-1]
        try:
            return [int(x) for x in reply.split(',')]
        except Exception as e:
            log.warning('Skipping text, because could not parse DeepSeek reply: {}'.format(orig_reply))
            return None

    def _score_single(self, args: tuple[list, str, str]) -> list:
        claims, input_text, answer = args
        q1 = self.prompt1(input_text, claims, answer)
        reply = self.chat.ask(q1)
        q2 = self.prompt2(input_text, claims, answer, reply)
        reply = self.chat.ask(q2)
        claim_labels: list[int] | None = self.parse_reply(reply)
        if claim_labels is None:
            return [np.nan for _ in range(len(claims))]  # will be skipped at evaluation
        if len(claim_labels) + 1 == len(claims):
            claim_labels.append(np.nan)  # last answer is undefined
        if len(claim_labels) != len(claims):
            log.warning(
                'Skipping text, because of inconsistend number of '
                'labels in {} reply: expected {}, got {}'.format(self.model, len(claims), reply))
            return [np.nan for _ in range(len(claims))]  # will be skipped at evaluation
        return [
            (
                np.nan if len(claims[i].aligned_token_ids) == 0 else
                1 if claim_labels[i] == 0 else
                0
            ) for i in range(len(claims))
        ]

    def __call__(
            self,
            stats: Dict[str, np.ndarray],
            target_texts: List[str] = None,
    ) -> list:
        input_texts = stats["input_texts"]

        if target_texts is None:
            if "answers" in stats.keys():
                target_texts = stats["answers"]
            elif "target_texts" in stats.keys():
                target_texts = stats["target_texts"]
            else:
                raise Exception("No answers or target_texts given")

        all_inputs = [
            (claims, input_text, answer)
            for input_text, claims, answer in zip(input_texts, stats["claims"], target_texts)
        ]

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(self._score_single, item) for item in all_inputs]
            claim_labels = []
            for future in tqdm(futures, desc="Verifying claims", disable=not self.progress_bar):
                claim_labels.append(future.result())

        return claim_labels
