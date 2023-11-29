import numpy as np

from typing import Dict

from .estimator import Estimator


def aggregate(posterior_mode, ue_name, token_level_data) -> np.ndarray:
    key = posterior_mode + "_token_level_scores"
    ue = token_level_data[key][ue_name]
    weights = token_level_data["weights"]

    return (weights * ue).sum(-1)


def all_token_estimators():
    return [
        PETtu(),
        PETdu(),
        PETmi(),
        PETrmi(),
        PETepkl(),
        PETent5(),
        PETent10(),
        PETent15(),
        EPTtu(),
        EPTdu(),
        EPTmi(),
        EPTrmi(),
        EPTepkl(),
        EPTent5(),
        EPTent10(),
        EPTent15(),
    ]


class EnsembleEstimator(Estimator):
    def __init__(self):
        super().__init__(["ensemble_token_scores"], "sequence")

    def __str__(self):
        raise NotImplementedError

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        stats["ensemble_token_scores"]

        ue = aggregate(
            self.posterior_mode, self.ue_name, stats["ensemble_token_scores"]
        )

        return np.array(ue)


class EPEnsembleEstimator(EnsembleEstimator):
    def __init__(self):
        self.posterior_mode = "ep"

        super().__init__()


class EPTtu(EPEnsembleEstimator):
    def __init__(self):
        self.ue_name = "total_uncertainty"

        super().__init__()

    def __str__(self):
        return "EPTtu"


class EPTdu(EPEnsembleEstimator):
    def __init__(self):
        self.ue_name = "data_uncertainty"

        super().__init__()

    def __str__(self):
        return "EPTdu"


class EPTmi(EPEnsembleEstimator):
    def __init__(self):
        self.ue_name = "mutual_information"

        super().__init__()

    def __str__(self):
        return "EPTmi"


class EPTrmi(EPEnsembleEstimator):
    def __init__(self):
        self.ue_name = "rmi"

        super().__init__()

    def __str__(self):
        return "EPTrmi"


class EPTepkl(EPEnsembleEstimator):
    def __init__(self):
        self.ue_name = "epkl"

        super().__init__()

    def __str__(self):
        return "EPTepkl"


class EPTent5(EPEnsembleEstimator):
    def __init__(self):
        self.ue_name = "entropy_top5"

        super().__init__()

    def __str__(self):
        return "EPTent5"


class EPTent10(EPEnsembleEstimator):
    def __init__(self):
        self.ue_name = "entropy_top10"

        super().__init__()

    def __str__(self):
        return "EPTent10"


class EPTent15(EPEnsembleEstimator):
    def __init__(self):
        self.ue_name = "entropy_top15"

        super().__init__()

    def __str__(self):
        return "EPTent15"


class PEEnsembleEstimator(EnsembleEstimator):
    def __init__(self):
        self.posterior_mode = "pe"

        super().__init__()


class PETtu(PEEnsembleEstimator):
    def __init__(self):
        self.ue_name = "total_uncertainty"

        super().__init__()

    def __str__(self):
        return "PETtu"


class PETdu(PEEnsembleEstimator):
    def __init__(self):
        self.ue_name = "data_uncertainty"

        super().__init__()

    def __str__(self):
        return "PETdu"


class PETmi(PEEnsembleEstimator):
    def __init__(self):
        self.ue_name = "mutual_information"

        super().__init__()

    def __str__(self):
        return "PETmi"


class PETrmi(PEEnsembleEstimator):
    def __init__(self):
        self.ue_name = "rmi"

        super().__init__()

    def __str__(self):
        return "PETrmi"


class PETepkl(PEEnsembleEstimator):
    def __init__(self):
        self.ue_name = "epkl"

        super().__init__()

    def __str__(self):
        return "PETepkl"


class PETent5(PEEnsembleEstimator):
    def __init__(self):
        self.ue_name = "entropy_top5"

        super().__init__()

    def __str__(self):
        return "PETent5"


class PETent10(PEEnsembleEstimator):
    def __init__(self):
        self.ue_name = "entropy_top10"

        super().__init__()

    def __str__(self):
        return "PETent10"


class PETent15(PEEnsembleEstimator):
    def __init__(self):
        self.ue_name = "entropy_top15"

        super().__init__()

    def __str__(self):
        return "PETent15"
