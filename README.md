[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/IINemo/lm-polygraph/blob/master/LICENSE.md)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Benchmark-yellow)](https://huggingface.co/LM-Polygraph)
<a href="https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00737/128713/Benchmarking-Uncertainty-Quantification-Methods" target="_blank"><img src=https://img.shields.io/badge/TACL-2025-blue.svg></a>
<a href="https://arxiv.org/pdf/2406.15627" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg></a>

# LM-Polygraph: Uncertainty estimation for LLMs

[Installation](#installation) | [Basic usage](#basic_usage) | [Overview](#overview_of_methods) | [Benchmark](#benchmark) | [Demo application](#demo_web_application) | [Documentation](https://lm-polygraph.readthedocs.io/)

LM-Polygraph provides a battery of state-of-the-art of uncertainty estimation (UE) methods for LMs in text generation tasks. High uncertainty can indicate the presence of hallucinations and knowing a score that estimates uncertainty can help to make applications of LLMs safer.

The framework also introduces an extendable benchmark for consistent evaluation of UE techniques by researchers and a demo web application that enriches the standard chat dialog with confidence scores, empowering end-users to discern unreliable responses.

## Installation

### From GitHub
The latest stable version is available in the main branch, it is recommended to use a virtual environment:

```shell
python -m venv env # Substitute this with your virtual environment creation command
source env/bin/activate
pip install git+https://github.com/IINemo/lm-polygraph.git
```

You can also use tags:

```shell
pip install git+https://github.com/IINemo/lm-polygraph.git@v0.5.0
```

### From PyPI
The latest tagged version is also available via PyPI:

```shell
pip install lm-polygraph
```

## <a name="basic_usage"></a>Basic usage
1. Initialize the base model (encoder-decoder or decoder-only) and tokenizer from HuggingFace or a local file, and use them to initialize the WhiteboxModel for evaluation:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_polygraph.utils.model import WhiteboxModel

model_path = "Qwen/Qwen2.5-0.5B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = WhiteboxModel(base_model, tokenizer, model_path=model_path)
```

2. Specify the UE method:
```python
from lm_polygraph.estimators import *

ue_method = MeanTokenEntropy()
```

3. Get predictions and their uncertainty scores:
```python
from lm_polygraph.utils.manager import estimate_uncertainty

input_text = "Who is George Bush?"
ue = estimate_uncertainty(model, ue_method, input_text=input_text)
print(ue)
# UncertaintyOutput(uncertainty=-6.504108926902215, input_text='Who is George Bush?', generation_text=' President of the United States', model_path='Qwen/Qwen2.5-0.5B-Instruct')
```

4. More examples: [basic_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/basic_example.ipynb)
5. See also a low-level example for efficient integration into your code: [low_level_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/low_level_example.ipynb)

## Using with LLMs deployed as a service

LM-Polygraph can work with any OpenAI-compatible API services:

```python
from lm_polygraph import BlackboxModel
from lm_polygraph.estimators import Perplexity, MaximumSequenceProbability

model = BlackboxModel.from_openai(
    openai_api_key='YOUR_API_KEY',
    model_path='gpt-4o',
    supports_logprobs=True  # Enable for deployments 
)

ue_method = Perplexity()  # or DetMat(), MeanTokenEntropy(), EigValLaplacian(), etc.
estimate_uncertainty(model, ue_method, input_text='What has a head and a tail but no body?')
```

UE methods such as `DetMat()` or `EigValLaplacian()` support fully blackbox LLMs that do not provide logits.

## More examples:

* [basic_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/basic_example.ipynb): simple examples of scoring individual queries
* [low_level_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/low_level_example.ipynb): low-level integration into inference and claim-level UE
* [low_level_vllm_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/low_level_vllm_example.ipynb): low-level example using vLLM for faster inference
* [basic_visual_llm_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/basic_visual_llm_example.ipynb): examples for visual LLMs

## <a name="overview_of_methods"></a>Overview of methods

<!-- | Uncertainty Estimation Method                                       | Type        | Category            | Compute | Memory | Need Training Data? |
| ------------------------------------------------------------------- | ----------- | ------------------- | ------- | ------ | ------------------- |
| Maximum sequence probability                                        | White-box   | Information-based   | Low     | Low    |         No          |
| Perplexity (Fomicheva et al., 2020a)                                | White-box   | Information-based   | Low     | Low    |         No          |
| Mean token entropy (Fomicheva et al., 2020a)                        | White-box   | Information-based   | Low     | Low    |         No          |
| Monte Carlo sequence entropy (Kuhn et al., 2023)                    | White-box   | Information-based   | High    | Low    |         No          |
| Pointwise mutual information (PMI) (Takayama and Arase, 2019)       | White-box   | Information-based   | Medium  | Low    |         No          |
| Conditional PMI (van der Poel et al., 2022)                         | White-box   | Information-based   | Medium  | Medium |         No          |
| Semantic entropy (Kuhn et al., 2023)                                | White-box   | Meaning diversity   | High    | Low    |         No          |
| Sentence-level ensemble-based measures (Malinin and Gales, 2020)    | White-box   | Ensembling          | High    | High   |         Yes         |
| Token-level ensemble-based measures (Malinin and Gales, 2020)       | White-box   | Ensembling          | High    | High   |         Yes         |
| Mahalanobis distance (MD) (Lee et al., 2018)                        | White-box   | Density-based       | Low     | Low    |         Yes         |
| Robust density estimation (RDE) (Yoo et al., 2022)                  | White-box   | Density-based       | Low     | Low    |         Yes         |
| Relative Mahalanobis distance (RMD) (Ren et al., 2023)              | White-box   | Density-based       | Low     | Low    |         Yes         |
| Hybrid Uncertainty Quantification (HUQ) (Vazhentsev et al., 2023a)  | White-box   | Density-based       | Low     | Low    |         Yes         |
| p(True) (Kadavath et al., 2022)                                     | White-box   | Reflexive           | Medium  | Low    |         No          |
| Number of semantic sets (NumSets) (Kuhn et al., 2023)               | Black-box   | Meaning Diversity   | High    | Low    |         No          |
| Sum of eigenvalues of the graph Laplacian (EigV) (Lin et al., 2023) | Black-box   | Meaning Diversity   | High    | Low    |         No          |
| Degree matrix (Deg) (Lin et al., 2023)                              | Black-box   | Meaning Diversity   | High    | Low    |         No          |
| Eccentricity (Ecc) (Lin et al., 2023)                               | Black-box   | Meaning Diversity   | High    | Low    |         No          |
| Lexical similarity (LexSim) (Fomicheva et al., 2020a)               | Black-box   | Meaning Diversity   | High    | Low    |         No          | -->

| Uncertainty Estimation Method                                                                                                                                                  | Type        | Category            | Compute | Memory | Need Training Data? | Level         |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ----------- | ------------------- |---------|--------| ------------------- |---------------|
| Maximum sequence probability                                                                                                                                                   | White-box   | Information-based   | Low     | Low    |         No          | sequence/claim |
| Perplexity [(Fomicheva et al., 2020a)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00330/96475/Unsupervised-Quality-Estimation-for-Neural-Machine)                  | White-box   | Information-based   | Low     | Low    |         No          | sequence/claim |
| Mean/max token entropy [(Fomicheva et al., 2020a)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00330/96475/Unsupervised-Quality-Estimation-for-Neural-Machine)      | White-box   | Information-based   | Low     | Low    |         No          | sequence/claim |
| Monte Carlo sequence entropy [(Kuhn et al., 2023)](https://openreview.net/forum?id=VD-AYtP0dve)                                                                                | White-box   | Information-based   | High    | Low    |         No          | sequence      |
| Pointwise mutual information (PMI) [(Takayama and Arase, 2019)](https://aclanthology.org/W19-4115/)                                                                            | White-box   | Information-based   | Medium  | Low    |         No          | sequence/claim |
| Conditional PMI [(van der Poel et al., 2022)](https://aclanthology.org/2022.emnlp-main.399/)                                                                                   | White-box   | Information-based   | Medium  | Medium |         No          | sequence      |
| Rényi divergence [(Darrin et al., 2023)](https://aclanthology.org/2023.emnlp-main.357/)                                                                                        | White-box   | Information-based   | Low     | Low    |         No          | sequence      |
| Fisher-Rao distance [(Darrin et al., 2023)](https://aclanthology.org/2023.emnlp-main.357/)                                                                                     | White-box   | Information-based   | Low     | Low    |         No          | sequence      |
| Attention Score [(Sriramanan et al., 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3c1e1fdf305195cd620c118aaa9717ad-Abstract-Conference.html)              | White-box   | Information-based   | Medium     | Low    |         No          | sequence/claim       |
| Focus [(Zhang et al., 2023)](https://aclanthology.org/2023.emnlp-main.58/)                                                                                                     | White-box   | Information-based   | Medium     | Low    |         No          | sequence/claim |
| Semantic entropy [(Kuhn et al., 2023)](https://openreview.net/forum?id=VD-AYtP0dve)                                                                                            | White-box   | Meaning diversity   | High    | Low    |         No          | sequence      |
| Claim-Conditioned Probability [(Fadeeva et al., 2024)](https://arxiv.org/abs/2403.04696)                                                                                       | White-box   | Meaning diversity   | Low     | Low    |         No          | sequence/claim |
| FrequencyScoring [(Mohri et al., 2024)](https://arxiv.org/abs/2402.10978)                                                                                                      | White-box   | Meaning diversity   | High    | Low    |         No          | claim |
| TokenSAR [(Duan et al., 2023)](https://arxiv.org/abs/2307.01379)                                                                                                               | White-box   | Meaning diversity   | High    | Low    |         No          | sequence/claim |
| SentenceSAR [(Duan et al., 2023)](https://arxiv.org/abs/2307.01379)                                                                                                            | White-box   | Meaning diversity   | High    | Low    |         No          | sequence      |
| SAR [(Duan et al., 2023)](https://arxiv.org/abs/2307.01379)                                                                                                                    | White-box   | Meaning diversity   | High    | Low    |         No          | sequence      |
| EigenScore [(Chen et al., 2024)](https://openreview.net/forum?id=Zj12nzlQbz)                                                                                                   | White-box   | Meaning diversity   | High    | Low    |         No          | sequence      |
| Sentence-level ensemble-based measures [(Malinin and Gales, 2020)](https://arxiv.org/abs/2002.07650)                                                                           | White-box   | Ensembling          | High    | High   |         Yes         | sequence      |
| Token-level ensemble-based measures [(Malinin and Gales, 2020)](https://arxiv.org/abs/2002.07650)                                                                              | White-box   | Ensembling          | High    | High   |         Yes         | sequence      |
| Mahalanobis distance (MD) [(Lee et al., 2018)](https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html)                                  | White-box   | Density-based       | Low     | Low    |         Yes         | sequence      |
| Robust density estimation (RDE) [(Yoo et al., 2022)](https://aclanthology.org/2022.findings-acl.289/)                                                                          | White-box   | Density-based       | Low     | Low    |         Yes         | sequence      |
| Relative Mahalanobis distance (RMD) [(Ren et al., 2023)](https://openreview.net/forum?id=kJUS5nD0vPB)                                                                          | White-box   | Density-based       | Low     | Low    |         Yes         | sequence      |
| Hybrid Uncertainty Quantification (HUQ) [(Vazhentsev et al., 2023a)](https://aclanthology.org/2023.acl-long.652/)                                                              | White-box   | Density-based       | Low     | Low    |         Yes         | sequence      |
| p(True) [(Kadavath et al., 2022)](https://arxiv.org/abs/2207.05221)                                                                                                            | White-box   | Reflexive           | Medium  | Low    |         No          | sequence/claim |
| Number of semantic sets (NumSets) [(Lin et al., 2023)](https://arxiv.org/abs/2305.19187)                                                                                       | Black-box   | Meaning Diversity   | High    | Low    |         No          | sequence      |
| Sum of eigenvalues of the graph Laplacian (EigV) [(Lin et al., 2023)](https://arxiv.org/abs/2305.19187)                                                                        | Black-box   | Meaning Diversity   | High    | Low    |         No          | sequence      |
| Degree matrix (Deg) [(Lin et al., 2023)](https://arxiv.org/abs/2305.19187)                                                                                                     | Black-box   | Meaning Diversity   | High    | Low    |         No          | sequence      |
| Eccentricity (Ecc) [(Lin et al., 2023)](https://arxiv.org/abs/2305.19187)                                                                                                      | Black-box   | Meaning Diversity   | High    | Low    |         No          | sequence      |
| Lexical similarity (LexSim) [(Fomicheva et al., 2020a)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00330/96475/Unsupervised-Quality-Estimation-for-Neural-Machine) | Black-box   | Meaning Diversity   | High    | Low    |         No          | sequence      |
| Kernel Language Entropy [(Nikitin et al., 2024)](https://arxiv.org/pdf/2405.20003)                                                                                             | Black-box | Meaning Diversity | High | Low | No | sequence      |
| LUQ [(Zhang et al., 2024)](https://aclanthology.org/2024.emnlp-main.299/)                                                                                                      | Black-box   | Meaning diversity   | High    | Low    |         No          | sequence      |
| Verbalized Uncertainty 1S [(Tian et al., 2023)](https://arxiv.org/abs/2305.14975)                                                                                              | Black-box   | Reflexive           | Low     | Low    |         No          | sequence      |
| Verbalized Uncertainty 2S [(Tian et al., 2023)](https://arxiv.org/abs/2305.14975)                                                                                              | Black-box   | Reflexive           | Medium  | Low    |         No          | sequence      |

## Benchmark

To evaluate the performance of uncertainty estimation methods consider a quick example:

```
CUDA_VISIBLE_DEVICES=0 polygraph_eval \
    --config-dir=./examples/configs/ \
    --config-name=polygraph_eval_coqa.yaml \
    model.path=meta-llama/Llama-3.1-8B \
    subsample_eval_dataset=100
```

To evaluate the performance of uncertainty estimation methods using vLLM for generation, consider the following example:

```
CUDA_VISIBLE_DEVICES=0 polygraph_eval \
    --config-dir=./examples/configs/ \
    --config-name=polygraph_eval_coqa.yaml \
    model=vllm \
    model.path=meta-llama/Llama-3.1-8B \
    estimators=default_estimators_vllm \
    stat_calculators=default_calculators_vllm \
    subsample_eval_dataset=100
```

You can also use a pre-built docker container for benchmarking, example:
```
docker run --gpus '"device=0"' --rm \
  -w /app \
  inemo/lm_polygraph \
  bash -c "polygraph_eval \
    --config-dir=./examples/configs/ \
    --config-name=polygraph_eval_coqa.yaml \
    model.path=meta-llama/Llama-3.1-8B \
    subsample_eval_dataset=100"
```

The benchmark datasets in the correct format could be found in the [HF repo](https://huggingface.co/LM-Polygraph). The scripts for dataset preparation could be found in the `dataset_builders` directory.

Use [`visualization_tables.ipynb`](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/vizualization_tables.ipynb) or [`result_tables.ipynb`](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/result_tables.ipynb) to generate the summarizing tables for an experiment.

A detailed description of the benchmark is in the [documentation](https://lm-polygraph.readthedocs.io/en/latest/usage.html#benchmarks).

## <a name="demo_web_application"></a>(Obsolete) Demo web application

Currently unsupported.

<img width="850" alt="gui7" src="https://github.com/IINemo/lm-polygraph/assets/21058413/51aa12f7-f996-4257-b1bc-afbec6db4da7">

## Cite

**TACL-2025:**
```
@article{shelmanovvashurin2025,
    author = {Vashurin, Roman and Fadeeva, Ekaterina and Vazhentsev, Artem and Rvanova, Lyudmila and Vasilev, Daniil and Tsvigun, Akim and Petrakov, Sergey and Xing, Rui and Sadallah, Abdelrahman and Grishchenkov, Kirill and Panchenko, Alexander and Baldwin, Timothy and Nakov, Preslav and Panov, Maxim and Shelmanov, Artem},
    title = {Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph},
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {13},
    pages = {220-248},
    year = {2025},
    month = {03},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00737},
    url = {https://doi.org/10.1162/tacl\_a\_00737},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00737/2511955/tacl\_a\_00737.pdf},
}
```

**EMNLP-2023 paper:**
```
@inproceedings{fadeeva-etal-2023-lm,
    title = "{LM}-Polygraph: Uncertainty Estimation for Language Models",
    author = "Fadeeva, Ekaterina  and
      Vashurin, Roman  and
      Tsvigun, Akim  and
      Vazhentsev, Artem  and
      Petrakov, Sergey  and
      Fedyanin, Kirill  and
      Vasilev, Daniil  and
      Goncharova, Elizaveta  and
      Panchenko, Alexander  and
      Panov, Maxim  and
      Baldwin, Timothy  and
      Shelmanov, Artem",
    editor = "Feng, Yansong  and
      Lefever, Els",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-demo.41",
    doi = "10.18653/v1/2023.emnlp-demo.41",
    pages = "446--461",
    abstract = "Recent advancements in the capabilities of large language models (LLMs) have paved the way for a myriad of groundbreaking applications in various fields. However, a significant challenge arises as these models often {``}hallucinate{''}, i.e., fabricate facts without providing users an apparent means to discern the veracity of their statements. Uncertainty estimation (UE) methods are one path to safer, more responsible, and more effective use of LLMs. However, to date, research on UE methods for LLMs has been focused primarily on theoretical rather than engineering contributions. In this work, we tackle this issue by introducing LM-Polygraph, a framework with implementations of a battery of state-of-the-art UE methods for LLMs in text generation tasks, with unified program interfaces in Python. Additionally, it introduces an extendable benchmark for consistent evaluation of UE techniques by researchers, and a demo web application that enriches the standard chat dialog with confidence scores, empowering end-users to discern unreliable responses. LM-Polygraph is compatible with the most recent LLMs, including BLOOMz, LLaMA-2, ChatGPT, and GPT-4, and is designed to support future releases of similarly-styled LMs.",
}
```

## Acknowledgements

The chat GUI implementation is based on the [chatgpt-web-application](https://github.com/ioanmo226/chatgpt-web-application) project.
