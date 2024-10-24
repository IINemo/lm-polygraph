---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{empty_brackets}
---

# Dataset Card for {dataset}

<!-- Provide a quick summary of the dataset. -->

This is a preprocessed version of {dataset} dataset for benchmarks in LM-Polygraph.

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

- **Curated by:** https://huggingface.co/LM-Polygraph
- **License:** https://github.com/IINemo/lm-polygraph/blob/main/LICENSE.md

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** https://github.com/IINemo/lm-polygraph

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

This dataset should be used for performing benchmarks on LM-polygraph.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

This dataset should not be used for further dataset preprocessing.

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

This dataset contains the "continuation" subset, which corresponds to main dataset, used in LM-Polygraph. It may also contain other subsets, which correspond to instruct methods, used in LM-Polygraph.

Each subset contains two splits: train and test. Each split contains two string columns: "input", which corresponds to processed input for LM-Polygraph, and "output", which corresponds to processed output for LM-Polygraph.

## Dataset Creation

### Curation Rationale

<!-- Motivation for the creation of this dataset. -->

This dataset is created in order to separate dataset creation code from benchmarking code.

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

Data is collected from {src_url} and processed by using build_dataset.py script in repository.

#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

People who created {src_url}

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

This dataset contains the same biases, risks, and limitations as its source dataset {src_url}

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users should be made aware of the risks, biases and limitations of the dataset.
