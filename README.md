[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/IINemo/lm-polygraph/blob/master/LICENSE.md)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
<a href="https://arxiv.org/pdf/2406.15627" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg></a>
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Benchmark-yellow)](https://huggingface.co/LM-Polygraph)

# LM-Polygraph: Uncertainty estimation for LLMs

[Installation](#installation) | [Basic usage](#basic_usage) | [Overview](#overview_of_methods) | [Benchmark](#benchmark) | [Demo application](#demo_web_application) | [Documentation](https://lm-polygraph.readthedocs.io/)

LM-Polygraph provides a battery of state-of-the-art of uncertainty estimation (UE) methods for LMs in text generation tasks. High uncertainty can indicate the presence of hallucinations and knowing a score that estimates uncertainty can help to make applications of LLMs safer.

The framework also introduces an extendable benchmark for consistent evaluation of UE techniques by researchers and a demo web application that enriches the standard chat dialog with confidence scores, empowering end-users to discern unreliable responses.

## Installation

### From GitHub
To install latest from main brach, clone the repo and conduct installation using pip, it is recommended to use virtual environment. Code example is presented below:

```shell
git clone https://github.com/IINemo/lm-polygraph.git
python3 -m venv env # Substitute this with your virtual environment creation command
source env/bin/activate
cd lm-polygraph
pip install .
```

Installation from GitHub is recommended if you want to explore notebooks with examples or use default benchmarking configurations, as they are included in the repository but not in the PyPI package. However code from the main branch may be unstable, so it is recommended to checkout to the latest stable release before installation:

```shell
git clone https://github.com/IINemo/lm-polygraph.git
git checkout tags/v0.3.0
python3 -m venv env # Substitute this with your virtual environment creation command
source env/bin/activate
cd lm-polygraph
pip install .
```

### From PyPI
To install the latest stable version from PyPI, run:

```shell
python3 -m venv env # Substitute this with your virtual environment creation command
source env/bin/activate
pip install lm-polygraph
```

To install a specific version, run:

```shell
python3 -m venv env # Substitute this with your virtual environment creation command
source env/bin/activate
pip install lm-polygraph==0.3.0
```

## <a name="basic_usage"></a>Basic usage
1. Initialize the base model (encoder-decoder or decoder-only) and tokenizer from HuggingFace or a local file, and use them to initialize the WhiteboxModel for evaluation. For example, with bigscience/bloomz-560m:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_polygraph.utils.model import WhiteboxModel

model_path = "bigscience/bloomz-560m"
base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = WhiteboxModel(base_model, tokenizer, model_path=model_path)
```

Alternatively, you can use ```WhiteboxModel.from_pretrained``` method to let LM-Polygraph download the model and tokenizer for you. However, this approach is deprecated and will be removed in the next major release.

```python
from lm_polygraph.utils.model import WhiteboxModel

model = WhiteboxModel.from_pretrained(
    "bigscience/bloomz-3b",
    device_map="cuda:0",
)
```

2. Specify UE method:

```python
from lm_polygraph.estimators import *

ue_method = MeanPointwiseMutualInformation()
```

3. Get predictions and their uncertainty scores:

```python
from lm_polygraph.utils.manager import estimate_uncertainty

input_text = "Who is George Bush?"
ue = estimate_uncertainty(model, ue_method, input_text=input_text)
print(ue)
# UncertaintyOutput(uncertainty=-6.504108926902215, input_text='Who is George Bush?', generation_text=' President of the United States', model_path='bigscience/bloomz-560m')
```

## Using Custom OpenAI-Compatible Endpoints for BlackBox Model Evaluation

To use LM-Polygraph with a custom OpenAI-compatible endpoint (like Azure OpenAI or self-hosted models), configure the following environment variables:

```shell
export OPENAI_BASE_URL="http://your-endpoint/v1"  # Your custom API endpoint
export OPENAI_API_KEY="your-api-key"              # Your API key
export OPENAI_API_TYPE="open_ai"                  # Use "open_ai" for OpenAI-compatible endpoints
```

This allows seamless integration with any OpenAI-compatible API service while maintaining the same interface and functionality.

## Greybox Models Support

LM-Polygraph supports "greybox" models - a middle ground between whitebox and blackbox models. Greybox models (like OpenAI models via API) provide access to token-level probabilities through logprobs while not requiring full model access. To use greybox capabilities with OpenAI models:

```python
from lm_polygraph import BlackboxModel
from lm_polygraph.estimators import Perplexity, MaximumSequenceProbability

# Create a model with greybox capabilities
model = BlackboxModel.from_openai(
    openai_api_key='YOUR_API_KEY',
    model_path='gpt-3.5-turbo',
    supports_logprobs=True  # Enable greybox capabilities
)

# Use information-based estimators normally only available to whitebox models
ue_method = Perplexity()  # or MaximumSequenceProbability(), MeanTokenEntropy(), etc.
```

With greybox support, you can use several information-based uncertainty estimators that would otherwise require full model access. Currently, the following estimators work in greybox mode in addition to standard blackbox methods:

- **MaximumSequenceProbability**
- **Perplexity**
- **MeanTokenEntropy**
- **ClaimConditionedProbability**

This significantly enhances uncertainty estimation options for API-based models.

### Other examples:

* [example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/basic_example.ipynb): simple examples of scoring individual queries;
* [claim_level_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/claim_level_example.ipynb): an example of scoring individual claims;
* [qa_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/qa_example.ipynb): an example of scoring the `bigscience/bloomz-3b` model on the `TriviaQA` dataset;
* [mt_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/mt_example.ipynb): an of scoring the `facebook/wmt19-en-de` model on the `WMT14 En-De` dataset;
* [ats_example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/ats_example.ipynb): an example of scoring the `facebook/bart-large-cnn` model on the `XSUM` summarization dataset;
* [colab](https://colab.research.google.com/drive/1JS-NG0oqAVQhnpYY-DsoYWhz35reGRVJ?usp=sharing): demo web application in Colab (`bloomz-560m`, `gpt-3.5-turbo`, and `gpt-4` fit the default memory limit; other models require Colab-pro).

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

Use [`visualization_tables.ipynb`](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/vizualization_tables.ipynb) or [`result_tables.ipynb`](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/result_tables.ipynb) to generate the summarizing tables for an experiment.

A detailed description of the benchmark is in the [documentation](https://lm-polygraph.readthedocs.io/en/latest/usage.html#benchmarks).

## <a name="demo_web_application"></a>Demo web application


<img width="850" alt="gui7" src="https://github.com/IINemo/lm-polygraph/assets/21058413/51aa12f7-f996-4257-b1bc-afbec6db4da7">


### Start with Docker

```sh
docker build -t lm-polygraph .
docker run --rm -it docker.io/library/lm-polygraph
# If you want to use GPU: docker run --gpus all --rm -it docker.io/library/lm-polygraph
```
The server should be available on `http://localhost:3001`

A more detailed description of the demo is available in the [documentation](https://lm-polygraph.readthedocs.io/en/latest/web_demo.html).

## Cite
EMNLP-2023 paper:
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

Submitted:
```
@article{vashurin2024benchmarking,
  title={Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph},
  author={
    Vashurin, Roman and
    Fadeeva, Ekaterina and
    Vazhentsev, Artem and
    Rvanova, Lyudmila and
    Tsvigun, Akim and
    Vasilev, Daniil and
    Xing, Rui and
    Sadallah, Abdelrahman Boda and
    Grishchenkov, Kirill and
    Petrakov, Sergey and
    Panchenko, Alexander and
    Baldwin, Timothy and
    Nakov, Preslav and
    Panov, Maxim and
    Shelmanov, Artem},
  journal={arXiv preprint arXiv:2406.15627},
  year={2024}
}
```

## Acknowledgements

The chat GUI implementation is based on the [chatgpt-web-application](https://github.com/ioanmo226/chatgpt-web-application) project.
