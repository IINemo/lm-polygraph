# Uncertainty for LLMs.

[Overleaf](https://www.overleaf.com/1371261441kstvywbbsnnj) с описанием методов.

[Вот тут](https://drive.google.com/drive/folders/1hHdr_dfAqzp0rkvezxsspZzXDuLQY4IY?usp=sharing) примеры работы token-level методов на задачах QA и MT.

* `main.py`: для запуска методов uncertainty
* `visualization.ipynb`: для генерации табличек с результатами

## Web application

### Installation
```
cd app && npm install && cd ../
```

### Starting the model server

Requires python3.10

```
python -m app.service
```

### Starting the web application server

```
node app/index.js
```

Once both servers are up and running, the chat model will be available at <http://localhost:3001/>.

----

## Старое

* `notebooks/CPMI_decoding_examples.ipynb`: примеры декодирования и оценки неопределенности с помощью CPMI [[1]](#1); примеры оценки неопределенности с помощью 5 методов из статьи Semantic Uncertanty [[2]](#2). Ноутбук плохо отображается в гитхабе, можно вместо этого смотреть в [колабе](https://colab.research.google.com/drive/1ZX0F-DSuo8WKNHewD7GZIq-4eI_kKiX4?usp=sharing).

* `notebooks/CPMI_decoding_TriviaQA_WMT.ipynb`: примеры работы тех-же методов декодирования на датасетах TriviaQA и WMT (ru-en). Оценка AUC-ROC разных методов (аналогично экспериментам из [[2]](#2)). Колаб [тут](https://colab.research.google.com/drive/1MFh256mU2g4bUshSRPr9Dclp6T1clv-d?usp=sharing).

## References
<a id="1">[1]</a> 
Mutual Information Alleviates Hallucinations in Abstractive Summarization, https://arxiv.org/pdf/2210.13210.pdf

<a id="2">[2]</a> 
Semantic Uncertanty: Linguistic Invariances for Uncertanty Estimation in Natural Language Generation, https://arxiv.org/pdf/2302.09664.pdf
