[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/IINemo/isanlp_srl_framebank/blob/master/LICENSE)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)

# LM-Polygraph: Uncertainty estimation for LLMs

[Installation](#installation) | [Basic usage](#basic_usage) | [Overview](#overview_of_methods) | [Benchmark](#benchmark) | [Demo application](#demo_web_application) | [Documentation](https://lm-polygraph.readthedocs.io/)

LM-Polygraph provides a battery of state-of-the-art of uncertainty estimation (UE) methods for LMs in text generation tasks. High uncertainty can indicate the presence of hallucinations and knowing a score that estimates uncertinaty can help to make applications of LLMs safer.

The framework also introduces an extendable benchmark for consistent evaluation of UE techniques by researchers and a demo web application that enriches the standard chat dialog with confidence scores, empowering end-users to discern unreliable responses.

## Installation

```
git clone https://github.com/IINemo/lm-polygraph.git && cd lm-polygraph && pip install .
```

## <a name="basic_usage"></a>Basic usage

1. Initialize the model (encoder-decoder or decoder-only) from HuggingFace or a local file. For example, `bigscience/bloomz-3b`
```python
from lm_polygraph.utils.model import WhiteboxModel

model = WhiteboxModel.from_pretrained(
    "bigscience/bloomz-3b",
    device="cuda:0",
)
```

2. Specify UE method
```python
from lm_polygraph.estimators import *

ue_method = MeanPointwiseMutualInformation()
```

3. Get predictions and their uncertainty scores
```python
from lm_polygraph.utils.manager import estimate_uncertainty

input_text = "Who is George Bush?"
estimate_uncertainty(model, ue_method, input_text=input_text)
```

### Other examples:

* [example.ipynb](https://github.com/IINemo/lm-polygraph/blob/main/examples/example.ipynb): simple examples of scoring individual queries;
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

| Uncertainty Estimation Method                                       | Type        | Category            | Compute | Memory | Need Training Data? |
| ------------------------------------------------------------------- | ----------- | ------------------- | ------- | ------ | ------------------- |
| Maximum sequence probability                                        | White-box   | Information-based   | Low     | Low    |         No          |
| Perplexity [(Fomicheva et al., 2020a)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00330/96475/Unsupervised-Quality-Estimation-for-Neural-Machine)                                | White-box   | Information-based   | Low     | Low    |         No          |
| Mean token entropy [(Fomicheva et al., 2020a)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00330/96475/Unsupervised-Quality-Estimation-for-Neural-Machine)                        | White-box   | Information-based   | Low     | Low    |         No          |
| Monte Carlo sequence entropy [(Kuhn et al., 2023)](https://openreview.net/forum?id=VD-AYtP0dve)                    | White-box   | Information-based   | High    | Low    |         No          |
| Pointwise mutual information (PMI) [(Takayama and Arase, 2019)](https://aclanthology.org/W19-4115/)       | White-box   | Information-based   | Medium  | Low    |         No          |
| Conditional PMI [(van der Poel et al., 2022)](https://aclanthology.org/2022.emnlp-main.399/)                         | White-box   | Information-based   | Medium  | Medium |         No          |
| Semantic entropy [(Kuhn et al., 2023)](https://openreview.net/forum?id=VD-AYtP0dve)                                | White-box   | Meaning diversity   | High    | Low    |         No          |
| Sentence-level ensemble-based measures [(Malinin and Gales, 2020)](https://arxiv.org/abs/2002.07650)    | White-box   | Ensembling          | High    | High   |         Yes         |
| Token-level ensemble-based measures [(Malinin and Gales, 2020)](https://arxiv.org/abs/2002.07650)       | White-box   | Ensembling          | High    | High   |         Yes         |
| Mahalanobis distance (MD) [(Lee et al., 2018)](https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html)                        | White-box   | Density-based       | Low     | Low    |         Yes         |
| Robust density estimation (RDE) [(Yoo et al., 2022)](https://aclanthology.org/2022.findings-acl.289/)                  | White-box   | Density-based       | Low     | Low    |         Yes         |
| Relative Mahalanobis distance (RMD) [(Ren et al., 2023)](https://openreview.net/forum?id=kJUS5nD0vPB)              | White-box   | Density-based       | Low     | Low    |         Yes         |
| Hybrid Uncertainty Quantification (HUQ) [(Vazhentsev et al., 2023a)](https://aclanthology.org/2023.acl-long.652/)  | White-box   | Density-based       | Low     | Low    |         Yes         |
| p(True) [(Kadavath et al., 2022)](https://arxiv.org/abs/2207.05221)                                     | White-box   | Reflexive           | Medium  | Low    |         No          |
| Number of semantic sets (NumSets) [(Kuhn et al., 2023)](https://openreview.net/forum?id=VD-AYtP0dve)               | Black-box   | Meaning Diversity   | High    | Low    |         No          |
| Sum of eigenvalues of the graph Laplacian (EigV) [(Lin et al., 2023)](https://arxiv.org/abs/2305.19187) | Black-box   | Meaning Diversity   | High    | Low    |         No          |
| Degree matrix (Deg) [(Lin et al., 2023)](https://arxiv.org/abs/2305.19187)                              | Black-box   | Meaning Diversity   | High    | Low    |         No          |
| Eccentricity (Ecc) [(Lin et al., 2023)](https://arxiv.org/abs/2305.19187)                               | Black-box   | Meaning Diversity   | High    | Low    |         No          |
| Lexical similarity (LexSim) [(Fomicheva et al., 2020a)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00330/96475/Unsupervised-Quality-Estimation-for-Neural-Machine)               | Black-box   | Meaning Diversity   | High    | Low    |         No          |




## Benchmark

To evaluate the performance of uncertainty estimation methods consider a quick example: 

```
HYDRA_CONFIG=../configs/polygraph_eval/polygraph_eval.yaml python ./scripts/polygraph_eval \
    dataset="./workdir/data/triviaqa.csv" \
    model="databricks/dolly-v2-3b" \
    save_path="./workdir/output" \
    seed=[1,2,3,4,5]
```

Use [`visualization_tables.ipynb`](https://github.com/IINemo/lm-polygraph/blob/main/notebooks/vizualization_tables.ipynb) to generate the summarizing tables for an experiment.

A detailed description of the benchmark is in the [documentation](https://lm-polygraph.readthedocs.io/en/latest/usage.html#benchmarks).

## <a name="demo_web_application"></a>Demo web application

 
<img width="850" alt="gui7" src="https://github.com/IINemo/lm-polygraph/assets/21058413/51aa12f7-f996-4257-b1bc-afbec6db4da7">


### Start with Docker

```sh
docker run -p 3001:3001 -it \
    -v $HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub \
    --gpus all mephodybro/polygraph_demo:0.0.17 polygraph_server
```
The server should be available on `http://localhost:3001`

A more detailed description of the demo is available in the [documentation](https://lm-polygraph.readthedocs.io/en/latest/web_demo.html).

## Cite
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
