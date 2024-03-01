import subprocess
import pathlib

import numpy as np

from lm_polygraph.utils.manager import UEManager
from lm_polygraph.estimators.ensemble_token_measures import all_token_estimators
from lm_polygraph.estimators.ensemble_sequence_measures import (
    all_ep_estimators,
    all_pe_estimators,
)


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


# ================= PE ensembles ==================


def test_pe_ensembles_dont_fail():
    exec_result = run_config_with_overrides(
        "test_polygraph_eval_ensemble", ensembling_mode="pe"
    )
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"


def test_pe_ensembles_has_all_ensemble_estimates():
    run_config_with_overrides("test_polygraph_eval_ensemble", ensembling_mode="pe")
    man = load_test_manager()

    expected_estimators = all_token_estimators() + all_pe_estimators()
    for estimator in expected_estimators:
        key = ("sequence", str(estimator))
        assert len(man.estimations[key]) > 0, f"result doesn't have {key}"


def test_pe_ensembles_no_nans():
    run_config_with_overrides("test_polygraph_eval_ensemble", ensembling_mode="pe")
    man = load_test_manager()

    expected_estimators = all_token_estimators() + all_pe_estimators()
    for estimator in expected_estimators:
        key = ("sequence", str(estimator))
        assert not (np.any(np.isnan(man.estimations[key]))), f"result has NaNs in {key}"


def test_pe_mi_not_zero():
    """
    If models are not the same, their output distributions should be different,
    and thus MI measures non-zero.
    """
    run_config_with_overrides("test_polygraph_eval_ensemble", ensembling_mode="pe")
    man = load_test_manager()

    mi_estimators = ["PETmi", "EPTmi", "PESrmi"]
    for estimator in mi_estimators:
        key = ("sequence", estimator)
        estimations = man.estimations[key]
        shape = len(estimations)
        assert not (
            np.allclose(estimations, np.zeros((shape,)), atol=1e-04)
        ), f"result has close to zero MI in {key}"


def test_pe_mi_zero_when_same():
    """
    If models are same, their output distributions should be identical,
    and thus MI measures equal to zero.
    """
    run_config_with_overrides(
        "test_polygraph_eval_ensemble", ensembling_mode="pe", mc_seeds="[42, 42]"
    )
    man = load_test_manager()

    mi_estimators = ["PETmi", "EPTmi", "PESrmi"]
    for estimator in mi_estimators:
        key = ("sequence", estimator)
        estimations = man.estimations[key]
        shape = len(estimations)
        assert np.allclose(
            estimations, np.zeros((shape,)), atol=1e-04
        ), f"result has non-zero MI in {key} when models are identical"


# ================= EP ensembles ==================


def test_ep_ensembles_dont_fail():
    exec_result = run_config_with_overrides(
        "test_polygraph_eval_ensemble", ensembling_mode="ep"
    )
    assert (
        exec_result.returncode == 0
    ), f"polygraph_eval returned code {exec_result.returncode} != 0"


def test_ep_ensembles_has_all_ensemble_estimates():
    run_config_with_overrides("test_polygraph_eval_ensemble", ensembling_mode="ep")
    man = load_test_manager()

    expected_estimators = all_token_estimators() + all_ep_estimators()
    for estimator in expected_estimators:
        key = ("sequence", str(estimator))
        assert len(man.estimations[key]) > 0, f"result doesn't have {key}"


def test_ep_ensembles_no_nans():
    run_config_with_overrides("test_polygraph_eval_ensemble", ensembling_mode="ep")
    man = load_test_manager()

    expected_estimators = all_token_estimators() + all_ep_estimators()
    for estimator in expected_estimators:
        key = ("sequence", str(estimator))
        assert not (np.any(np.isnan(man.estimations[key]))), f"result has NaNs in {key}"


def test_ep_mi_not_zero():
    """
    If models are not the same, their output distributions should be different,
    and thus MI measures non-zero.
    """
    run_config_with_overrides("test_polygraph_eval_ensemble", ensembling_mode="ep")
    man = load_test_manager()

    mi_estimators = ["PETmi", "EPTmi", "EPSrmi"]
    for estimator in mi_estimators:
        key = ("sequence", estimator)
        estimations = man.estimations[key]
        shape = len(estimations)
        assert not (
            np.allclose(estimations, np.zeros((shape,)), atol=1e-04)
        ), f"result has close to zero MI in {key}"


def test_ep_mi_zero_when_same():
    """
    If models are same, their output distributions should be identical,
    and thus MI measures equal to zero.
    """
    run_config_with_overrides(
        "test_polygraph_eval_ensemble", ensembling_mode="ep", mc_seeds="[42, 42]"
    )
    man = load_test_manager()

    mi_estimators = ["PETmi", "EPTmi", "EPSrmi"]
    for estimator in mi_estimators:
        key = ("sequence", estimator)
        estimations = man.estimations[key]
        shape = len(estimations)
        assert np.allclose(
            estimations, np.zeros((shape,)), atol=1e-04
        ), f"result has non-zero MI in {key} when models are identical"
