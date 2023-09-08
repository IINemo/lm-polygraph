import os
import subprocess


def exec_bash(s):
    return subprocess.run(s, shell=True)


def test_just_works_cli():
    command = 'scripts/polygraph_eval --estimator LexicalSimilarity --estimator_kwargs \'{"metric": "rouge2"}\' --dataset SpeedOfMagic/trivia_qa_tiny --model bigscience/bloomz-560m --no-ignore_exceptions --subsample_eval_dataset 10'
    exec_result = exec_bash(command)
    assert exec_result.returncode == 0, f'polygraph_eval returned code {exec_result.returncode} != 0'
    os.remove('_seed1')


def test_just_works_hydra():
    command = 'HYDRA_CONFIG=../test/configs/test_polygraph_eval.yaml scripts/polygraph_eval'
    exec_result = exec_bash(command)
    assert exec_result.returncode == 0, f'polygraph_eval returned code {exec_result.returncode} != 0'
    os.remove('_seed1')
