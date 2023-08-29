from typing import List
from factscore.atomic_facts import AtomicFactGenerator
from lm_polygraph.token_clusterer.token_clusterer import TokenClusterer
from lm_polygraph.token_clusterer.matcher import Matcher


class AtomicClusterer(TokenClusterer):
    def __init__(self, matcher: Matcher, reduction: str = 'mean'):
        super().__init__(reduction)
        self._af_generator = AtomicFactGenerator(
            key_path='/Users/ekaterinafadeeva/Downloads/openai_key.txt',
            demon_dir='/Users/ekaterinafadeeva/Downloads/demos',
            gpt3_cache_file='/Users/ekaterinafadeeva/Downloads/InstructGPT.pkl')
        self._matcher = matcher

    def cluster(self, text: str, tokens: List[str]) -> List[int]:
        curr_afs, _ = self._af_generator.run(text)
        match_strings = [self._matcher(sent, sent_atoms) for sent, sent_atoms in curr_afs]
        last = 0
        new_class = 0
        letter_classes = []
        for (sent, _), matches in zip(curr_afs, match_strings):
            while not text[last:].lower().startswith(sent.lower()):
                last += 1
                letter_classes += [None]
            cur_classes = [None for _ in sent]
            for match_str in matches:
                assert len(match_str) == len(sent)
                for i in range(len(sent)):
                    if match_str[i] == '^':
                        assert cur_classes[i] is None
                        cur_classes[i] = new_class
                new_class += 1
            letter_classes += cur_classes
            last += len(cur_classes)
        letter_classes += [None for _ in range(len(text) - last)]
        return letter_classes
