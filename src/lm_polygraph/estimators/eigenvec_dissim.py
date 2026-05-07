class EigVecDissimilarity(Estimator):
    def __init__(
            self,
            similarity_score: Literal["NLI_score", "Jaccard_score"] = "NLI_score",
            affinity: Literal["entail", "contra"] = "entail",  # relevant for NLI score case
            verbose: bool = False,
            thres: float = 0.9,
            samples_source: str = "sample",
    ):
        if not samples_source.startswith('greedy+'):
            samples_source = 'greedy+' + samples_source
        if similarity_score == "NLI_score":
            if affinity == "entail":
                super().__init__([f"{samples_source}_semantic_matrix_entail", f"{samples_source}_texts"], "sequence")
            else:
                super().__init__([f"{samples_source}_semantic_matrix_contra", f"{samples_source}_texts"], "sequence")
        else:
            super().__init__([f"{samples_source}_texts"], "sequence")

        self.similarity_score = similarity_score
        self.affinity = affinity
        self.verbose = verbose
        self.thres = thres
        self.samples_source = samples_source

    def __str__(self):
        base = "EigVecDissimilarity"
        if self.samples_source != "sample":
            base += f'_{self.samples_source}'
        if self.similarity_score == "NLI_score":
            return f"{base}_{self.similarity_score}_{self.affinity}"
        return f"{base}_{self.similarity_score}"

    def U_Eccentricity(self, i, stats):
        answers = stats[f"{self.samples_source}_texts"][i]

        if self.similarity_score == "NLI_score":
            if self.affinity == "entail":
                W = np.array(stats[f"{self.samples_source}_semantic_matrix_entail"])[i, :, :]
            else:
                W = 1 - np.array(stats[f"{self.samples_source}_semantic_matrix_contra"])[i, :, :]
            W = (W + np.transpose(W)) / 2
        else:
            W = compute_sim_score(
                answers=answers,
                affinity=self.affinity,
                similarity_score=self.similarity_score,
            )

        D = np.diag(W.sum(axis=1))
        D_inverse_sqrt = np.linalg.inv(np.sqrt(D))
        L = np.eye(D.shape[0]) - D_inverse_sqrt @ W @ D_inverse_sqrt

        # k is hyperparameter  - Number of smallest eigenvectors to retrieve
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(L)

        if self.thres is not None:
            keep_mask = eigenvalues < self.thres
            eigenvalues, smallest_eigenvectors = (
                eigenvalues[keep_mask],
                eigenvectors[:, keep_mask],
            )
        else:
            smallest_eigenvectors = eigenvectors

        smallest_eigenvectors = smallest_eigenvectors.T

        C_Ecc = np.mean([
            np.linalg.norm(smallest_eigenvectors[:, i] - smallest_eigenvectors[:, 0]) ** 2
            for i in range(1, smallest_eigenvectors.shape[1])
        ])
        return C_Ecc

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        res = []
        for i, answers in enumerate(stats[f"{self.samples_source}_texts"]):
            if self.verbose:
                log.debug(f"generated answers: {answers}")
            conf = self.U_Eccentricity(i, stats)
            res.append(conf)
        return np.array(res)


class EigVecDissimilarityP(Estimator):
    def __init__(
            self,
            similarity_score: Literal["NLI_score", "Jaccard_score"] = "NLI_score",
            affinity: Literal["entail", "contra"] = "entail",  # relevant for NLI score case
            verbose: bool = False,
            thres: float = 0.9,
            samples_source: str = "beamsearch",
            **process_probs_args,
    ):
        if not samples_source.startswith('greedy+'):
            samples_source = 'greedy+' + samples_source
        if similarity_score == "NLI_score":
            if affinity == "entail":
                super().__init__([
                    f"{samples_source}_log_likelihoods",
                    f"{samples_source}_semantic_matrix_entail",
                    f"{samples_source}_texts",
                ], "sequence")
            else:
                super().__init__([
                    f"{samples_source}_log_likelihoods",
                    f"{samples_source}_semantic_matrix_contra",
                    f"{samples_source}_texts",
                ], "sequence")
        else:
            super().__init__([
                f"{samples_source}_log_likelihoods",
                f"{samples_source}_texts",
            ], "sequence")

        self.similarity_score = similarity_score
        self.affinity = affinity
        self.verbose = verbose
        self.thres = thres
        self.samples_source = samples_source
        self.process_probs_args = process_probs_args

    def __str__(self):
        base = "EigVecDissimilarityP"
        if self.samples_source != "sample":
            base += f'_{self.samples_source}'
        if self.similarity_score == "NLI_score":
            return f"{base}_{self.similarity_score}_{self.affinity}"
        return f"{base}_{self.similarity_score}"

    def U_Eccentricity(self, i, stats):
        answers = stats[f"{self.samples_source}_texts"][i]

        if self.similarity_score == "NLI_score":
            if self.affinity == "entail":
                W = np.array(stats[f"{self.samples_source}_semantic_matrix_entail"])[i, :, :]
            else:
                W = 1 - np.array(stats[f"{self.samples_source}_semantic_matrix_contra"])[i, :, :]
            W = (W + np.transpose(W)) / 2
        else:
            W = compute_sim_score(
                answers=answers,
                affinity=self.affinity,
                similarity_score=self.similarity_score,
            )

        D = np.diag(W.sum(axis=1))
        D_inverse_sqrt = np.linalg.inv(np.sqrt(D))
        L = np.eye(D.shape[0]) - D_inverse_sqrt @ W @ D_inverse_sqrt

        # k is hyperparameter  - Number of smallest eigenvectors to retrieve
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(L)

        if self.thres is not None:
            keep_mask = eigenvalues < self.thres
            eigenvalues, smallest_eigenvectors = (
                eigenvalues[keep_mask],
                eigenvectors[:, keep_mask],
            )
        else:
            smallest_eigenvectors = eigenvectors

        smallest_eigenvectors = smallest_eigenvectors.T

        sample_token_lls = stats[f"{self.samples_source}_log_likelihoods"][i]
        probs = np.array([np.exp(sum(s)) for s in sample_token_lls])
        probs = probs[1:]
        probs = process_probs(probs, **self.process_probs_args)

        C_Ecc = (np.array([
            np.linalg.norm(smallest_eigenvectors[:, i] - smallest_eigenvectors[:, 0]) ** 2
            for i in range(1, smallest_eigenvectors.shape[1])
        ]) * probs).sum()

        return C_Ecc

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        res = []
        for i, answers in enumerate(stats[f"{self.samples_source}_texts"]):
            if self.verbose:
                log.debug(f"generated answers: {answers}")
            conf = self.U_Eccentricity(i, stats)
            res.append(conf)
        return np.array(res)