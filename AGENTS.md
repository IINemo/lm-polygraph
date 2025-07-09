 # Onboarding Guide for New Contributors

 Welcome to **LM-Polygraph**! This guide provides an overview of the library structure, main components, and benchmarking workflow to help new contributors get started.

 ## Library Structure

 The repository follows a standard layout:

 ```text
 .
 ├── scripts/                 # Entry point scripts (e.g., polygraph_eval)
 ├── src/lm_polygraph/        # Core modules: stat calculators, estimators, generation metrics, UE manager, etc.
 ├── examples/                # Example Hydra configs for benchmarking
 ├── tests/                   # Unit and integration tests
 ├── docs/                    # Additional documentation
 ├── CONTRIBUTING.md          # Contribution guidelines
 ├── README.md                # Installation and quick start
 └── pyproject.toml           # Project metadata and dependencies
 ```

 ## Main Components

 ### Stat Calculators

 StatCalculators compute intermediate statistics required by uncertainty estimators and metrics. They are implemented under `src/lm_polygraph/stat_calculators/`.

 ```python
 # src/lm_polygraph/stat_calculators/stat_calculator.py
 class StatCalculator(ABC):
     @abstractmethod
     def __call__(
         self,
         dependencies: Dict[str, np.ndarray],
         texts: List[str],
         model: Model,
         max_new_tokens: int = 100,
     ) -> Dict[str, np.ndarray]:
         ...
 ```
 【F:src/lm_polygraph/stat_calculators/stat_calculator.py†L9-L53】

 Default calculators are registered via `register_default_stat_calculators`:

 ```python
 # src/lm_polygraph/defaults/register_default_stat_calculators.py
 def register_default_stat_calculators(...):
     ...
 ```
 【F:src/lm_polygraph/defaults/register_default_stat_calculators.py†L1-L60】

 ### Estimators

 Estimators consume precomputed stats to output uncertainty scores. They live under `src/lm_polygraph/estimators/`.

 ```python
 # src/lm_polygraph/estimators/estimator.py
 class Estimator(ABC):
     @abstractmethod
     def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
         ...
 ```
 【F:src/lm_polygraph/estimators/estimator.py†L8-L50】

 ### Generation Metrics

 GenerationMetrics compute quality scores (e.g., ROUGE, BLEU, BERTScore) by comparing model outputs to references. They live under `src/lm_polygraph/generation_metrics/`.

 ```python
 # src/lm_polygraph/generation_metrics/generation_metric.py
 class GenerationMetric(ABC):
     @abstractmethod
     def __call__(
         self,
         stats: Dict[str, np.ndarray],
         target_texts: List[str]
     ) -> np.ndarray:
         ...
 ```
 【F:src/lm_polygraph/generation_metrics/generation_metric.py†L8-L60】

 **Preprocessing & multi-reference support**:
 - `PreprocessOutputTarget` applies custom preprocessing to outputs and targets before computing a base metric.
 - `AggregatedMetric` wraps a base metric to handle multiple references per sample via aggregation (e.g., max over all references).

 ```python
 # preprocess_output_target.py
 # aggregated_metric.py
 ```
 【F:src/lm_polygraph/generation_metrics/preprocess_output_target.py†L1-L51】【F:src/lm_polygraph/generation_metrics/aggregated_metric.py†L1-L57】

 ### Model Wrappers

LM-Polygraph provides two main model wrapper types in `src/lm_polygraph/utils/model.py`:

**BlackboxModel** (no direct access to logits)  
A client for remote inference APIs (OpenAI or Hugging Face Inference).  
- Factory methods: `from_openai` (chat/completions with optional logprobs) and `from_huggingface` (HF inference endpoint).  
- `_validate_args`: drops unsupported HF-generate arguments and maps HF parameter names (e.g., `max_new_tokens` → `max_tokens`).  
- `generate_texts`: loops over prompts, retries on API errors, returns generated text (and optionally `logprobs`, `tokens` when `supports_logprobs=True`).  
- Raises if attempting to use logits-related features on unsupported backends.  

```python
# src/lm_polygraph/utils/model.py (BlackboxModel)
class BlackboxModel(Model):
    ...
```
【F:src/lm_polygraph/utils/model.py†L73-L260】

**WhiteboxModel** (direct Hugging Face integration)  
Wraps a HF `AutoModelForCausalLM`/`AutoModelForSeq2SeqLM` and `AutoTokenizer`, exposing raw logits and custom generation hooks.  
- `from_pretrained`: loads model & tokenizer, infers architecture (causal vs seq2seq), sets `pad_token`.  
- `_ScoresProcessor`: intercepts HF’s `logits_processor` to capture original `log_softmax` scores before any processing.  
- `_MultiTokenEOSCriteria`: custom multi-token EOS/stop-sequence stopping criterion.  
- `generate`: merges default `GenerationParameters` with call-site overrides, attaches `StoppingCriteriaList` & `_ScoresProcessor`, invokes `model.generate`, and swaps in raw scores.  
- `tokenize`: applies optional chat template, batch-pads inputs, returns `input_ids`/`attention_mask`.  
- `generate_texts`: decodes only newly generated tokens (or full seq for seq2seq), stripping prompts when needed.  
- `create_ensemble`: supports Monte Carlo dropout-based ensembling via `EnsembleGenerationMixin`.  

```python
# src/lm_polygraph/utils/model.py (WhiteboxModel)
class WhiteboxModel(Model):
    ...
```
【F:src/lm_polygraph/utils/model.py†L380-L735】

 ## Benchmarking Pipeline (`polygraph_eval`)

All experiments start from `scripts/polygraph_eval`:

 ```bash
 #!/usr/bin/env python3
 @hydra.main(...)
 def main(args):
     ...

 if __name__ == "__main__":
     main()
 ```
 【F:scripts/polygraph_eval†L1-L12】

High-level workflow in `main`:

 ```text
 for seed in args.seed:
     model = get_model(args)
    # Load model & data
    model = get_model(args)                     # dispatches to WhiteboxModel/BlackboxModel/vLLM
    dataset = Dataset.load(...)                 # supports CSV and HF datasets, returns (input_texts, target_texts) batches

    # Build components
    stat_calcs = get_stat_calculator_names(args) # containers for StatCalculator factories (handles 'auto')
    estimators = get_ue_methods(args, model)    # Instantiates Estimator classes from config
    gen_metrics = get_generation_metrics(args)   # Sets up GenerationMetric instances, wrapping preprocessors & multi-ref
    ue_metrics = get_ue_metrics(args)            # Metrics for evaluating UE quality (e.g. PRAUC, ROCAUC)

     manager = UEManager(
         data=dataset,
         model=model,
         estimators=estimators,
         builder_env_stat_calc=BuilderEnvironmentStatCalculator(model),
         available_stat_calculators=stat_calcs,
         generation_metrics=gen_metrics,
         ue_metrics=ue_metrics,
         processors=[Logger()],
         ...
     )
    manager()                                   # run generation, stat calc, estimation, metrics, and correlations
    manager.save(...)                           # save results (metrics, estimations, stats) per seed
 ```
 【F:scripts/polygraph_eval†L131-L169】

### Configuration via Hydra

LM-Polygraph leverages Hydra to compose benchmarking presets. Configs live under `examples/configs/` (standard) and `examples/configs/instruct/` (instruction-tuned).

Key features:
1. **`defaults`** section to select model, estimators, stat_calculators, dataset, and base processing hooks.
2. Overrides for task-specific fields: `dataset`, `task`, `n_shot`, `batch_size`, `generation_params`, etc.
3. Optional `process_output_fn`/`process_target_fn` scripts and `multiref` flag to enable custom preprocessing and multi-reference aggregation.

```yaml
# examples/configs/polygraph_eval_coqa.yaml
hydra:
  run:
    dir: ${cache_path}/${task}/${model.path}/${dataset}/${now:%Y-%m-%d}/${now:%H-%M-%S}
defaults:
  - model: bloomz-560m
  - estimators: default_estimators
  - stat_calculators: default_calculators
  - base_processing_coqa
  - _self_
# ...
```
【F:examples/configs/polygraph_eval_coqa.yaml†L1-L20】

 ### StatCalculator Selection & Ordering

The set and execution order of StatCalculators is determined by the statistics required by your Estimators and GenerationMetrics, plus mandatory fields like `greedy_texts` (and `greedy_tokens` for Whitebox models).

1. **`get_stat_calculator_names(config)`** (scripts/polygraph_eval):
   - Reads `config.stat_calculators`. If `"auto"` is present, registers default calculators (with flags for attentions/hidden_states/logprobs).
   - Allows custom calculators via YAML entries (`name`, `cfg`, `stats`, `dependencies`, `builder`).

2. **`UEManager.init()`** (src/lm_polygraph/utils/manager.py):
   - Builds `stat_calculators_dict` mapping each stat key to its container.
   - Collects dependencies dictionary from each container.
   - Aggregates required stats: all `e.stats_dependencies` (from Estimators) + `m.stats_dependencies` (from GenerationMetrics) + `greedy_*`.
   - Calls `order_calculators`, which topologically sorts the stat keys, ensuring dependencies are satisfied.
   - Filters out redundant blackbox variants and ensemble-specific stats.
   - Instantiates actual StatCalculator instances via `FactoryStatCalculator`.

```python
# scripts/polygraph_eval.py: get_stat_calculator_names
# src/lm_polygraph/utils/manager.py: order_calculators & UEManager.init
```
【F:scripts/polygraph_eval†L170-L204】【F:src/lm_polygraph/utils/manager.py†L61-L88】【F:src/lm_polygraph/utils/manager.py†L186-L233】

 ### UEManager Workflow

The `UEManager` orchestrates the core benchmarking loop:

**1. `calculate(batch_stats, ...)`**  
   - Iterates through ordered StatCalculator instances.  
   - Each calculator transforms and appends new stats to `batch_stats`.  
   - Automatically duplicates stats for blackbox prefixes when needed.  
   - Catches and logs exceptions (optional skip via `ignore_exceptions`), allowing downstream Estimators to fail gracefully.

**2. `estimate(batch_stats, ...)`**  
   - Calls each `Estimator` with the accumulated `batch_stats`.  
   - Converts results to lists, flattens claim-level results, and collects them in `self.estimations`.  
   - Removes any estimators that error on a batch, tracking `total_bad_estimators`.

**3. Batch callback (`__call__`)**  
   - Runs `calculate` → `estimate` → generation metrics on each batch.  
   - Applies `generation_metric(batch_stats, target_texts)`, converting to lists and claim-level flattening.  
   - Records `greedy_texts`/`greedy_tokens` and invokes optional `Processor.on_batch`.

**4. Final correlation metrics**  
   - After all batches, for each pair of (generation metric, UE metric) and matching `level`, computes:  
     - Oracle & random baselines via `ue_metric(-proxy, proxy)` & `get_random_scores`.  
     - Filters NaN pairs, then computes the raw UE metric and a normalized score.  
   - Logs warnings if any NaNs occur in estimators or generation metrics.

```python
# src/lm_polygraph/utils/manager.py: calculate, estimate, __call__
```
【F:src/lm_polygraph/utils/manager.py†L240-L350】【F:src/lm_polygraph/utils/manager.py†L350-L445】

 #### Handling NaN Values

LM-Polygraph ensures robust correlation computation by filtering out invalid entries:
- `_delete_nans(ue, metric)` uses `np.nan_to_num` to replace estimator NaNs/∞ with large finite sentinels, then masks out positions where the ground-truth metric is NaN.
- If no valid pairs remain (`len(ue)==0`), the final metric is recorded as NaN.

```python
def _delete_nans(ue, metric):
    metric = np.asarray(metric)
    # Clip estimator NaN/∞ to finite sentinel values
    clipped_ue = np.nan_to_num(ue, nan=-1e7, neginf=-1e7, posinf=1e7)
    # Drop any positions where ground-truth metric is NaN
    mask = ~np.isnan(metric)
    return clipped_ue[mask], metric[mask]
```
【F:src/lm_polygraph/utils/manager.py†L48-L56】

Warnings are emitted if any NaNs are detected in estimator outputs or generation metrics prior to filtering:
【F:src/lm_polygraph/utils/manager.py†L425-L445】

 ## Uncertainty Evaluation Metrics

Uncertainty Evaluation (UE) metrics assess how well your uncertainty estimators align with ground-truth generation errors. Key components are defined in `src/lm_polygraph/ue_metrics/ue_metric.py`:

```python
def normalize(target: List[float]) -> np.ndarray: ...
def skip_target_nans(target, estimator) -> Tuple[List, List]: ...
class UEMetric(ABC): ...
def get_random_scores(function, metrics, num_iter=1000, seed=42) -> float: ...
def normalize_metric(target_score, oracle_score, random_score) -> float: ...
```
【F:src/lm_polygraph/ue_metrics/ue_metric.py†L7-L24】【F:src/lm_polygraph/ue_metrics/ue_metric.py†L26-L57】【F:src/lm_polygraph/ue_metrics/ue_metric.py†L60-L75】

- **`UEMetric`**: base class requiring a unique `__str__()` and `__call__(estimator, target) -> float`.  
- **`get_random_scores`**: random baseline by shuffling estimator indices, averaged over many trials.  
- **`normalize_metric`**: rescales a raw UE metric between the random baseline (0) and oracle baseline (1).  
- **`skip_target_nans`**: in-metric helper to drop samples where the ground-truth metric is NaN before computing quality.

### Prediction-Rejection Area (PRR)

The most widely used UE metric is **Prediction-Rejection Area (PRR)**, implemented in `pred_rej_area.py`.
PRR constructs a **prediction–rejection curve** that tracks the average ground-truth metric over the retained subset as samples with highest uncertainty are dropped incrementally up to a fraction `max_rejection`.

Raw PRR is computed as the (discrete) area under this curve (integral from 0 to `max_rejection`), normalized by the number of rejection steps. To compare methods, the final PRR score is further normalized between two baselines:
1. **Oracle curve** (using ground-truth metric itself as uncertainty scores) – the optimal upper bound.
2. **Random curve** (random rejection, a flat line at the global mean) – the lower bound.

Final normalized PRR:
```
(raw_prr - random_prr) / (oracle_prr - random_prr)
```

```python
# src/lm_polygraph/ue_metrics/pred_rej_area.py
class PredictionRejectionArea(UEMetric):
    def __init__(self, max_rejection: float = 1.0):
        super().__init__()
        self.max_rejection = max_rejection

    def __str__(self):
        return "prr" if self.max_rejection == 1 else f"prr_{self.max_rejection}"

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        target = normalize(target)
        ue = np.array(estimator)
        num_obs = len(ue)
        num_rej = int(self.max_rejection * num_obs)
        ue_argsort = np.argsort(ue)
        sorted_metrics = np.array(target)[ue_argsort]
        cumsum = np.cumsum(sorted_metrics)[-num_rej:]
        scores = (cumsum / np.arange((num_obs - num_rej) + 1, num_obs + 1))[::-1]
        return np.sum(scores) / num_rej
```
【F:src/lm_polygraph/ue_metrics/pred_rej_area.py†L13-L21】【F:src/lm_polygraph/ue_metrics/pred_rej_area.py†L22-L25】【F:src/lm_polygraph/ue_metrics/pred_rej_area.py†L27-L39】【F:src/lm_polygraph/ue_metrics/pred_rej_area.py†L40-L53】

### Saving Results

At the end of each seed run, results are saved to disk:

```python
# scripts/polygraph_eval.py (finally block)
try:
    man()
finally:
    man.save(save_path + f"/ue_manager_seed{seed}")
```
【F:scripts/polygraph_eval†L147-L154】

Here `save_path` usually comes from Hydra (`save_path: ${hydra:run.dir}`), producing files like:
```
<run_dir>/ue_manager_seed1
<run_dir>/ue_manager_seed2
...
```

Each file is a PyTorch checkpoint containing a dict with keys:
```python
# src/lm_polygraph/utils/manager.py
def save(self, save_path: str):
    torch.save(
        {
            "state": self.state,
            "metrics": self.metrics,
            "gen_metrics": self.gen_metrics,
            "estimations": self.estimations,
            "stats": self.stats,
        },
        save_path,
    )
```
【F:src/lm_polygraph/utils/manager.py†L459-L476】

You can reload these results with `UEManager.load(load_path)` to inspect past runs or resume analysis.

Welcome aboard, and happy benchmarking!

-------

After implementing changes, run linters:

```bash
# Run linters
black .
flake8 --extend-ignore E501,F405,F403,E203 --per-file-ignores __init__.py:F401,builder_stat_calculator_simple.py:F401 .
```
