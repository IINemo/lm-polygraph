# Reproducible CPU experiments

`cpu-py312.txt` pins the complete dependency resolution for Linux x86_64,
Python 3.12, CPU PyTorch, and the `evaluation`, `openai`, and `dev` extras.
Constraints restrict versions; they do not install unused extras. The minimal
installation uses the same file without pulling in evaluation or API packages.

From the repository root, in a fresh environment:

```bash
python3.12 -m venv .venv-experiment
source .venv-experiment/bin/activate
python -m pip install --upgrade pip
python -m pip install torch -c constraints/cpu-py312.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[evaluation,openai,dev]' -c constraints/cpu-py312.txt
python -m pip check
python -m pytest test/unit -q
```

For the minimal profile, replace `.[evaluation,openai,dev]` with `.[dev]`.
The installation workflow checks both profiles with offline numerical, import,
metadata, missing-dependency, CSV, lexical-similarity, and client-construction
tests. It does not download model weights or call hosted APIs. The constraints
are a tested installation baseline, not a claim that every estimator/model
combination has been validated.

The CPU pins are not intended for CUDA, vLLM, COMET, the legacy demo, or
bitsandbytes environments. Resolve those separately for the hardware and model
stack you use. Also pin model/tokenizer revisions, dataset revisions, generation
settings, and seeds in experiments; a package constraints file cannot capture
those inputs.

To refresh this profile using Python 3.12 and `pip-tools`:

```bash
pip-compile pyproject.toml --extra evaluation --extra openai --extra dev \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --strip-extras --allow-unsafe --no-annotate --no-emit-index-url --no-emit-trusted-host \
  --output-file constraints/cpu-py312.txt
```

Existing pins are retained where possible. To deliberately update PyTorch,
select a CPU build explicitly with `--upgrade-package 'torch==VERSION+cpu'`.
Review the diff and repeat both clean installation checks before committing.
