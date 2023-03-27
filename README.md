# Uncertainty for LLMs.

* `CPMI_decoding_examples.ipynb`: примеры декодирования и оценки неопределенности с помощью CPMI [[1]](#1); примеры оценки неопределенности с помощью 5 методов из статьи Semantic Uncertanty [[2]](#2). Ноутбук плохо отображается в гитхабе, можно вместо этого смотреть в [колабе](https://colab.research.google.com/drive/1ZX0F-DSuo8WKNHewD7GZIq-4eI_kKiX4?usp=sharing).

* `CPMI_decoding_TriviaQA_WMT.ipynb`: примеры работы тех-же методов декодирования на датасетах TriviaQA и WMT (ru-en). Оценка AUC-ROC разных методов (аналогично экспериментам из [[2]](#2)). Колаб [тут](https://colab.research.google.com/drive/1MFh256mU2g4bUshSRPr9Dclp6T1clv-d?usp=sharing).

## References
<a id="1">[1]</a> 
Mutual Information Alleviates Hallucinations in Abstractive Summarization, https://arxiv.org/pdf/2210.13210.pdf

<a id="2">[2]</a> 
Semantic Uncertanty: Linguistic Invariances for Uncertanty Estimation in Natural Language Generation, https://arxiv.org/pdf/2302.09664.pdf
