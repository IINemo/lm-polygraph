import subprocess
import pathlib
import pytest
import time
from openai import OpenAI

from lm_polygraph.utils.manager import UEManager

# from lm_polygraph.estimators.ensemble_token_measures import all_token_estimators
# from lm_polygraph.estimators.ensemble_sequence_measures import (
#    all_ep_estimators,
#    all_pe_estimators,
# )


# ================= TEST HELPERS ==================


def exec_bash(s):
    return subprocess.run(s, shell=True)


def pwd():
    return pathlib.Path(__file__).parent.resolve()


def load_test_manager():
    return UEManager.load(f"{pwd()}/../workdir/output/test/ue_manager_seed1")


def run_config_with_overrides(config_name, **overrides):
    command = f"HYDRA_CONFIG={pwd()}/configs/{config_name}.yaml polygraph_eval"
    for key, value in overrides.items():
        command += f" {key}='{value}'"
    print(command)
    exec_result = exec_bash(command)

    assert exec_result.returncode == 0, f"running {command} failed!"

    return exec_result


# ================= TEST CASES ==================


def test_just_works():
    exec_result = run_config_with_overrides("test_polygraph_eval")
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"


def test_all_seq_ue():
    exec_result = run_config_with_overrides("test_polygraph_eval_seq_ue")
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"


def test_chat_completion(mock_response, request):  # Added request parameter
    """Test chat completion with mock server"""
    client = OpenAI()
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test message"}],
                temperature=0.7,
                max_tokens=50,
            )

            content = response.choices[0].message.content.strip()
            assert content == mock_response, "Unexpected response content"
            break

        except Exception as e:
            if attempt == max_retries - 1:
                pytest.fail(f"All attempts failed: {str(e)}")
            time.sleep(retry_delay)


def test_blackbox_local():
    exec_result = run_config_with_overrides("test_polygraph_eval_blackbox")
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"


# ================= PE ensembles ==================
#
#
# def test_pe_ensembles_dont_fail():
#     overrides = {
#         "model.ensembling_mode": "pe",
#     }
#     exec_result = run_config_with_overrides("test_polygraph_eval_ensemble", **overrides)
#     assert (
#         exec_result.returncode == 0
#     ), f"polygraph_eval returned code {exec_result.returncode} != 0"
#
#
# def test_pe_ensembles_has_all_ensemble_estimates():
#     overrides = {
#         "model.ensembling_mode": "pe",
#     }
#     run_config_with_overrides("test_polygraph_eval_ensemble", **overrides)
#     man = load_test_manager()
#
#     expected_estimators = all_token_estimators() + all_pe_estimators()
#     for estimator in expected_estimators:
#         key = ("sequence", str(estimator))
#         assert len(man.estimations[key]) > 0, f"result doesn't have {key}"
#
#
# def test_pe_ensembles_no_nans():
#     overrides = {
#         "model.ensembling_mode": "pe",
#     }
#     run_config_with_overrides("test_polygraph_eval_ensemble", **overrides)
#     man = load_test_manager()
#
#     expected_estimators = all_token_estimators() + all_pe_estimators()
#     for estimator in expected_estimators:
#         key = ("sequence", str(estimator))
#         assert not (np.any(np.isnan(man.estimations[key]))), f"result has NaNs in {key}"
#
#
# def test_pe_mi_not_zero():
#     """
#     If models are not the same, their output distributions should be different,
#     and thus MI measures non-zero.
#     """
#     overrides = {
#         "model.ensembling_mode": "pe",
#     }
#     run_config_with_overrides("test_polygraph_eval_ensemble", **overrides)
#     man = load_test_manager()
#
#     mi_estimators = ["PETmi", "EPTmi", "PESrmi"]
#     for estimator in mi_estimators:
#         key = ("sequence", estimator)
#         estimations = man.estimations[key]
#         shape = len(estimations)
#         assert not (
#             np.allclose(estimations, np.zeros((shape,)), atol=1e-04)
#         ), f"result has close to zero MI in {key}"
#
#
# def test_pe_mi_zero_when_same():
#     """
#     If models are same, their output distributions should be identical,
#     and thus MI measures equal to zero.
#     """
#     overrides = {
#         "model.ensembling_mode": "pe",
#         "model.mc_seeds": "[42, 42]",
#     }
#     run_config_with_overrides("test_polygraph_eval_ensemble", **overrides)
#
#     man = load_test_manager()
#
#     mi_estimators = ["PETmi", "EPTmi", "PESrmi"]
#     for estimator in mi_estimators:
#         key = ("sequence", estimator)
#         estimations = man.estimations[key]
#         shape = len(estimations)
#         assert np.allclose(
#             estimations, np.zeros((shape,)), atol=1e-04
#         ), f"result has non-zero MI in {key} when models are identical"
#
#
# # ================= EP ensembles ==================
#
#
# def test_ep_ensembles_dont_fail():
#     overrides = {
#         "model.ensembling_mode": "ep",
#     }
#     exec_result = run_config_with_overrides("test_polygraph_eval_ensemble", **overrides)
#     assert (
#         exec_result.returncode == 0
#     ), f"polygraph_eval returned code {exec_result.returncode} != 0"
#
#
# def test_ep_ensembles_has_all_ensemble_estimates():
#     overrides = {
#         "model.ensembling_mode": "ep",
#     }
#     run_config_with_overrides("test_polygraph_eval_ensemble", **overrides)
#     man = load_test_manager()
#
#     expected_estimators = all_token_estimators() + all_ep_estimators()
#     for estimator in expected_estimators:
#         key = ("sequence", str(estimator))
#         assert len(man.estimations[key]) > 0, f"result doesn't have {key}"
#
#
# def test_ep_ensembles_no_nans():
#     overrides = {
#         "model.ensembling_mode": "ep",
#     }
#     run_config_with_overrides("test_polygraph_eval_ensemble", **overrides)
#     man = load_test_manager()
#
#     expected_estimators = all_token_estimators() + all_ep_estimators()
#     for estimator in expected_estimators:
#         key = ("sequence", str(estimator))
#         assert not (np.any(np.isnan(man.estimations[key]))), f"result has NaNs in {key}"
#
#
# def test_ep_mi_not_zero():
#     """
#     If models are not the same, their output distributions should be different,
#     and thus MI measures non-zero.
#     """
#     overrides = {
#         "model.ensembling_mode": "ep",
#     }
#     run_config_with_overrides("test_polygraph_eval_ensemble", **overrides)
#     man = load_test_manager()
#
#     mi_estimators = ["PETmi", "EPTmi", "EPSrmi"]
#     for estimator in mi_estimators:
#         key = ("sequence", estimator)
#         estimations = man.estimations[key]
#         shape = len(estimations)
#         assert not (
#             np.allclose(estimations, np.zeros((shape,)), atol=1e-04)
#         ), f"result has close to zero MI in {key}"
#
#
# def test_ep_mi_zero_when_same():
#     """
#     If models are same, their output distributions should be identical,
#     and thus MI measures equal to zero.
#     """
#     overrides = {
#         "model.ensembling_mode": "ep",
#         "model.mc_seeds": "[42, 42]",
#     }
#     run_config_with_overrides("test_polygraph_eval_ensemble", **overrides)
#     man = load_test_manager()
#
#     mi_estimators = ["PETmi", "EPTmi", "EPSrmi"]
#     for estimator in mi_estimators:
#         key = ("sequence", estimator)
#         estimations = man.estimations[key]
#         shape = len(estimations)
#         assert np.allclose(
#             estimations, np.zeros((shape,)), atol=1e-04
#         ), f"result has non-zero MI in {key} when models are identical"
