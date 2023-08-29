# import string
# import numpy as np
#
# from typing import List
# from abc import ABC, abstractmethod
# from keybert import KeyBERT
#
#
# class Matcher(ABC):
#     # Should return number of match_str for each atom. They should not intersect.
#     @abstractmethod
#     def __call__(self, text: str, atoms: List[str]) -> List[str]:
#         # Example:
#         #   text = 'Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida.',
#         # atoms = ['Lanny Flaherty is an American actor.',
#         #          'Lanny Flaherty was born on December 18, 1949.',
#         #          'Lanny Flaherty was born in Pensacola, Florida.']
#         #  return ['                     ^^^^^^^^ ^^^^^                                                  ',
#         #          '                                    ^^^^ ^^ ^^^^^^^^ ^^  ^^^^                        ',
#         #          '                                                                  ^^^^^^^^^  ^^^^^^^ ']
#         raise Exception('Not implemented')
#
#
# class KeyWordMatcher(Matcher):
#     def __init__(self, thrs: float = 0.2, maximums_difference_coef: float = 1.5, verbose: bool = False):
#         self._kw_model = KeyBERT()
#         self._thrs = thrs
#         self._maximums_difference_coef = maximums_difference_coef
#         self._verbose = verbose
#
#     def __call__(self, text: str, atoms: List[str]) -> List[str]:
#         kw_res = self._kw_model.extract_keywords(atoms, top_n=max(len(s) for s in atoms + ['x']))
#         if len(atoms) == 1:
#             kw_res = [kw_res]
#
#         if self._verbose:
#             print(f'Match')
#             print(f'\ttext: {text}')
#             print('\tatoms:')
#             for a in atoms:
#                 print(f'\t\t{a}')
#             print('Keywords:')
#             for k in kw_res:
#                 print('\t', [kw for kw, _ in k])
#
#         words = [w.strip(string.punctuation) for w in text.split()]
#         last = 0
#         match_strings = [[' ' for _ in text] for _ in atoms]
#         for w in words:
#             while not text[last:].lower().startswith(w.lower()):
#                 last += 1
#             values, idx = [], []
#             for i in range(len(atoms)):
#                 for kw, val in kw_res[i]:
#                     if kw.lower() == w.lower():
#                         values.append(val)
#                         idx.append(i)
#             if len(values) != 0:
#                 best = np.max(values)
#                 accept = True
#                 if len(values) >= 2:
#                     best2 = np.partition(values, -2)[-2]
#                     if best < self._maximums_difference_coef * best2:
#                         accept = False
#                 if accept:
#                     a = idx[np.argmax(values)]
#                     match_strings[a][last:last + len(w)] = ['^' for _ in w]
#             last += len(w)
#
#         if self._verbose:
#             print('Matches:')
#             for ms in match_strings:
#                 print()
#                 print(text)
#                 print(''.join(ms))
#
#         return [''.join(s) for s in match_strings]
