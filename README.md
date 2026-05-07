# Don't Throw Away Your Beams

## Improving Consistency-Based Uncertainties in LLMs via Beam Search

<a href="https://arxiv.org/abs/2512.09538" target="_blank"><img src="https://img.shields.io/badge/arXiv-2512.09538-b31b1b.svg"></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/IINemo/lm-polygraph/blob/master/LICENSE.md)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)

Official implementation of the ICLR 2026 paper:

**Don't Throw Away Your Beams: Improving Consistency-Based Uncertainties in LLMs via Beam Search**

> **TL;DR:** Beam search improves consistency-based uncertainty estimation in LLMs by producing diverse, high-probability candidate answers with lower variance than multinomial sampling.

---

## Overview

Consistency-based uncertainty estimation methods usually generate multiple candidate answers and measure agreement between them. These methods often rely on multinomial sampling.

However, in short-form question answering, multinomial sampling can produce:

- duplicate generations,
- unstable uncertainty estimates,
- high variance across runs.

We show that replacing multinomial sampling with **beam search** improves consistency-based uncertainty estimation. In particular, beam search produces candidate answers that are more diverse, high-probability, and stable.

Our beam-weighted estimators achieve state-of-the-art results across:

- 6 QA benchmarks,
- 6 LLMs,
- multiple consistency-based uncertainty estimation methods.

<p align="center">
  <img width="850" src="https://github.com/IINemo/lm-polygraph/blob/beam-uncertainty/assets/beamsearch.pdf">
</p>

---

## Main Results

The table below reports average **PRR ↑** over six QA datasets. Beam-search variants consistently improve over their multinomial-sampling counterparts.

| Method | Llama 3.1 8B Base | Llama 3.1 8B Instruct | Gemma 3 4B Base | Gemma 3 4B Instruct | Qwen3 8B Base | Qwen3 8B Instruct |
|---|---:|---:|---:|---:|---:|---:|
| Dissimilarity | .505 | .379 | .630 | .206 | .477 | .327 |
| **Dissimilarity + Beam Search** | **.543 ↑** | **.417 ↑** | **.650 ↑** | **.252 ↑** | **.478 ↑** | **.355 ↑** |
| Eccentricity | .453 | .368 | .563 | .231 | .396 | .251 |
| **Eccentricity + Beam Search** | **.505 ↑** | **.397 ↑** | **.603 ↑** | **.285 ↑** | **.410 ↑** | **.345 ↑** |
| EigVecDissimilarity | .463 | .370 | .561 | .236 | .425 | .256 |
| **EigVecDissimilarity + Beam Search** | **.510 ↑** | **.414 ↑** | **.598 ↑** | **.301 ↑** | **.450 ↑** | **.376 ↑** |
| CoCoA-MSP | .505 | .404 | .587 | .314 | .461 | .334 |
| **CoCoA-MSP + Beam Search** | **.521 ↑** | **.426 ↑** | **.615 ↑** | **.345 ↑** | **.473 ↑** | **.347 ↑** |
| CoCoA-PPL | .523 | .397 | .628 | .312 | .461 | .327 |
| **CoCoA-PPL + Beam Search** | **.536 ↑** | **.412 ↑** | **.649 ↑** | **.339 ↑** | .461 | **.337 ↑** |

---

## Installation

We recommend using a virtual environment.

```bash
python -m venv env
source env/bin/activate

pip install git+https://github.com/IINemo/lm-polygraph.git@beam-uncertainty
```

---

## Running Benchmarks

Run an evaluation with:

```bash
polygraph_eval \
    --config-dir=./examples/configs/ \
    --config-name=polygraph_eval_coqa.yaml \
    model.path=meta-llama/Llama-3.1-8B \
    estimators=beamsearch
```

You can change the dataset using `--config-name` and the model using `model.path`.

---

## Datasets

| Dataset | Base LLM Config | Instruct LLM Config |
|---|---|---|
| TriviaQA | `polygraph_eval_triviaqa.yaml` | `polygraph_eval_triviaqa_instruct.yaml` |
| WebQuestions | `polygraph_eval_webq.yaml` | `polygraph_eval_webq_instruct.yaml` |
| CoQA | `polygraph_eval_coqa.yaml` | `polygraph_eval_coqa_instruct.yaml` |
| HotpotQA | `polygraph_eval_hotpotqa.yaml` | `polygraph_eval_hotpotqa_instruct.yaml` |
| CommonsenseQA | `polygraph_eval_csqa.yaml` | `polygraph_eval_csqa_instruct.yaml` |
| ARC-Challenge | `polygraph_eval_arcchallenge.yaml` | `polygraph_eval_arcchallenge_instruct.yaml` |

---

## Supported Models

| Model | Benchmark Argument |
|---|---|
| Llama 3.1 8B Base | `model.path=meta-llama/Llama-3.1-8B` |
| Llama 3.1 8B Instruct | `model.path=meta-llama/Llama-3.1-8B-Instruct` |
| Gemma 3 4B Base | `model=gemma_3 model.path=google/gemma-3-4b-pt` |
| Gemma 3 4B Instruct | `model=gemma_3 model.path=google/gemma-3-4b-it` |
| Qwen3 8B Base | `model.path=Qwen/Qwen3-8B-Base` |
| Qwen3 8B Instruct | `model.path=Qwen/Qwen3-8B` |

---

## Implemented Methods

This branch implements beam-search variants of several consistency-based uncertainty estimators.

| Method | Import |
|---|---|
| Dissimilarity + Beam Search | `from lm_polygraph.estimators import DissimilarityP` |
| Eccentricity + Beam Search | `from lm_polygraph.estimators import EccentricityPConf` |
| EigVecDissimilarity + Beam Search | `from lm_polygraph.estimators import EigVecDissimilarityP` |
| CoCoA + Beam Search | `from lm_polygraph.estimators import CocoaMSPP, CocoaPPLP, CocoaMTEP` |

This implementation also supports LM-Polygraph baselines, including:

- Semantic Entropy,
- SAR,
- Lexical Similarity,
- P(True),
- Perplexity,
- CCP,
- and others.

---

## Citation

If you use this code, please cite:

```bibtex
@misc{fadeeva2025dontthrowawaybeams,
      title={Don't Throw Away Your Beams: Improving Consistency-Based Uncertainties in LLMs via Beam Search}, 
      author={Ekaterina Fadeeva and Maiya Goloburda and Aleksandr Rubashevskii and Roman Vashurin and Artem Shelmanov and Preslav Nakov and Mrinmaya Sachan and Maxim Panov},
      year={2025},
      eprint={2512.09538},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2512.09538}
}
```

---

## Acknowledgements

This work builds on top of the [LM-Polygraph](https://github.com/IINemo/lm-polygraph) framework.
