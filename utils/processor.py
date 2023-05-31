import numpy as np

from typing import List, Dict, Tuple


class Processor:
    def on_batch(
            self,
            batch_stats: Dict[str, np.ndarray],
            batch_gen_metrics: Dict[Tuple[str, str], List[float]],
            batch_estimations: Dict[Tuple[str, str], List[float]],
    ):
        pass

    def on_eval(self, metrics: Dict[Tuple[str, str, str, str], float]):
        pass


class Logger(Processor):
    def on_batch(
            self,
            batch_stats: Dict[str, np.ndarray],
            batch_gen_metrics: Dict[Tuple[str, str], List[float]],
            batch_estimations: Dict[Tuple[str, str], List[float]],
    ):
        print('=' * 50 + ' NEW BATCH ' + '=' * 50)
        print('Statistics:')
        print()
        for key, val in batch_stats.items():
            str_repr = str(val)
            # to skip large outputs
            if len(str_repr) < 10000 and str_repr.count('\n') < 10:
                print(f'{key}: {val}')
                print()
        print('-' * 100)
        print('Estimations:')
        print()
        for key, val in batch_estimations.items():
            print(f'{key}: {val}')
            print()
        print('-' * 100)
        print('Generation metrics:')
        print()
        for key, val in batch_gen_metrics.items():
            print(f'{key}: {val}')
            print()

    def on_eval(self, metrics: Dict[Tuple[str, str, str, str], float]):
        print('=' * 50 + ' METRICS ' + '=' * 50)
        print('Metrics:')
        print()
        for key, val in metrics.items():
            print(f'{key}: {val}')
            print()
