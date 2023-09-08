import subprocess


def exec_bash(s):
    return subprocess.run(s, shell=True)


# Check that launching script works and all required files are created
def test_just_works_cli():
    command = 'scripts/polygraph_eval --estimator LexicalSimilarity --estimator_kwargs \'{"metric": "rouge2"}\' --dataset SpeedOfMagic/trivia_qa_tiny --model databricks/dolly-v2-3b'
    exec_result = exec_bash(command)
    assert exec_result.returncode == 0, f'polygraph_eval returned code {exec_result.returncode} != 0'


# def test_just_works_hydra():
#     assert TEST_SCRIPT is not None, 'TEST_SCRIPT env variable was not found'  # ./test_imdb_bert-base-uncased_42.sh
#     workdir = get_workdir()

#     exec_bash(f'rm -r {workdir}')
#     if LAUNCH_SCRIPT == 'true':
#         print('Launching test script...')
#         exec_bash(TEST_SCRIPT)
#         print('Done launching test script')
#     assert os.path.exists(workdir), f'Resulting directory {workdir} is missing'

#     assert os.path.exists(workdir / 'stat.tsv'), 'stat.tsv file is missing'
#     assert os.path.exists(workdir / 'log.txt'), 'log.txt file is missing'
#     assert os.path.exists(workdir / 'training_args.json'), 'training_args.json is not found'
#     assert os.path.exists(workdir / 'eval_dataset'), 'eval_dataset is not found'
#     for beam in ['beam_0.0', 'beam_0.1', 'beam_1.0', 'beam_1.1']:
#         beam_dir = workdir / beam
#         assert os.path.exists(beam_dir), 'Beam not found'
#         assert os.path.exists(beam_dir / 'model'), 'Model in beam is not found'
#         assert os.path.exists(beam_dir / 'train_sample.json'), 'Train samples in beam are not found'
#         assert os.path.exists(beam_dir / 'embeddings.torch'), 'embeddings.torch is not found'
#         assert os.path.exists(beam_dir / 'idx.txt'), 'idx.txt is not found'
#         assert os.path.exists(beam_dir / 'next_possible_queries.txt'), 'next_possible_queries.txt is not found'
#         assert os.path.exists(beam_dir / 'all_possible_queries.txt'), 'all_possible_queries.txt is not found'
#         assert os.path.exists(beam_dir / 'prev_beam.txt'), 'prev_beam.txt is not found'
#         assert os.path.exists(beam_dir / 'metric.json'), 'metric.json is not found'
#         assert os.path.exists(beam_dir / 'ue.json'), 'ue.json is not found'
