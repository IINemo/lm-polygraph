# Dataset Builder

This program is used to build datasets used in benchmarks in LM-Polygraph.

## Requirements

All requirements will be installed when installing lm-polygraph.

## Usage

The main entry point is `manager.py`, that can be launched by the following command:

```bash
python3 manager.py
```

It supports the following options:

* `--dataset {DATASET}` - choose dataset to build. You can see all available datasets by passing option `-h` or `--help`.
* `-s`, `--save-to-disk` - save built dataset to `result/{DATASET}` folder. You can retrieve it by using `datasets.load_from_disk` function.
* `-p`, `--publish` - publish dataset to Hugging Face. It requires `HF_TOKEN` environment variable to be set.
* `-n`, `--namespace` - namespace, where dataset will be located in Hugging Face. Default value is `LM-Polygraph`.

## Directory contents

* `manager.py` - entry point for dataset builder.
* `build_dataset.py` - code that builds dataset by its name and returns `datasets.Dataset` object.
* `builders` - code and configs for building specific datasets. For example, `builders/mmlu.py` contains code for building `mmlu` dataset and its instruct variations.
* `template_readme.md` - Template for dataset card for generated datasets.

## Instructions

### How to build and save existing dataset to disk.

Suppose your dataset is `{DATASET}`. Then, run the following command:

```bash
python3 manager.py --dataset {DATASET} --save-to-disk
```

### How to build and publish existing dataset to Hugging Face

Suppose your dataset is `{DATASET}`, and you are publishing to namespace LM-Polygraph (default namespace). Then, run the following command:

```
export HF_TOKEN={your hf token}
python3 manager.py --dataset {DATASET} --publish
```

Note the following:
*  If this dataset was never published before, then new repository will be created and its dataset card will be created automatically.

* If this dataset was already published, then dataset will be updated, **but dataset card will remain the same**.
