# import re
# import string
#
# from keybert import KeyBERT
# from typing import List
# from lm_polygraph.token_clusterer.token_clusterer import TokenClusterer
#
#
# class KeyWordsClusterer(TokenClusterer):
#     def __init__(self, reduction: str = 'mean', thrs: float = 0.1):
#         super().__init__(reduction)
#         self._thrs = thrs
#         self._kw_model = KeyBERT()
#
#     def cluster(self, text: str, tokens: List[str]) -> List[int]:
#         pieces = [p.strip() for p in re.split('\n|\.|\?|\!', text) if len(p.strip()) > 0]
#         kw_res = self._kw_model.extract_keywords(pieces, top_n=max(len(s) for s in pieces + ['x']))
#         if len(pieces) == 1:
#             kw_res = [kw_res]
#         kw_res = [[(word, val) for word, val in piece_res if val > self._thrs] for piece_res in kw_res]
#         last = 0
#         marker_str = ''
#         for i in range(len(pieces)):
#             while not text[last:].startswith(pieces[i]):
#                 last += 1
#                 marker_str += ' '
#             words = [w.strip(string.punctuation) for w in pieces[i].split()]
#             for w in words:
#                 while not text[last:].startswith(w):
#                     last += 1
#                     marker_str += ' '
#                 last += len(w)
#                 if w.lower() in [w for w, _ in kw_res[i]]:
#                     marker_str += '^' * len(w)
#                 else:
#                     marker_str += ' ' * len(w)
#         marker_str += ' ' * (len(text) - last)
#         #       text: Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida.
#         # marker_str: ^^^^^ ^^^^^^^^       ^^^^^^^^ ^^^^^ ^^^^    ^^^^^^^^ ^^  ^^^^     ^^^^^^^^^  ^^^^^^^
#         last, new_class = 0, 0
#         classes: List[int] = []
#         for t in tokens:
#             if t in ['<s>', '</s>']:
#                 classes.append(None)
#                 continue
#             tok = t.strip()
#             while not text[last:].startswith(tok):
#                 last += 1
#             if marker_str[last:last + len(tok)] == '^' * len(tok):
#                 classes.append(new_class)
#                 new_class += 1
#             else:
#                 classes.append(None)
#             last += len(tok)
#
#         #       text: Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida.
#         #    classes:   0      1     n  n      2      3    4   n      5    6    7    n      8         9
#         return classes
