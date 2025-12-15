# Metrics Explained

The `polygraph_eval` script computes two main types of metrics:

1. **Generation Metrics** (`gen_metrics`) - Measure how good the model's generated text is
2. **Uncertainty Estimation Metrics** (`ue_metrics`) - Measure how well uncertainty estimators correlate with generation quality

## 1. Generation Metrics (Ground Truth Quality)

These metrics compare the model's generated text to the ground truth (reference) text. **Higher values = better quality**.

### Available Generation Metrics:

#### **ROUGE Metrics** (`RougeMetric`)
- **ROUGE-1**: Overlap of unigrams (single words) between generated and reference
- **ROUGE-2**: Overlap of bigrams (word pairs) between generated and reference  
- **ROUGE-L**: Longest common subsequence between generated and reference
- **Range**: 0-1 (normalized by reference)
- **Use case**: Text summarization, question answering

#### **BLEU** (`BLEUMetric`)
- Measures n-gram precision between generated and reference
- **Range**: 0-1
- **Use case**: Machine translation, text generation

#### **BERTScore** (`BertScoreMetric`)
- Semantic similarity using BERT embeddings
- **Range**: 0-1
- **Use case**: General text quality assessment

#### **SBERT** (`SbertMetric`)
- Semantic similarity using Sentence-BERT embeddings
- **Range**: 0-1
- **Use case**: Semantic similarity measurement

#### **BARTScore** (`BartScoreSeqMetric`)
- Quality score using BART model
- **Range**: Negative values (higher is better)
- **Use case**: Summarization quality

#### **AlignScore** (`AlignScore`)
- Factual consistency and alignment score
- **Range**: 0-1
- **Use case**: Factual accuracy in QA/summarization

#### **COMET** (`Comet`)
- Translation quality using COMET model
- **Range**: 0-1
- **Use case**: Machine translation

#### **Accuracy** (`AccuracyMetric`)
- Exact match accuracy
- **Range**: 0-1
- **Use case**: Classification tasks, exact answer matching

### How They're Used:
- These metrics serve as **ground truth uncertainty** - if generation quality is low, that's "high uncertainty"
- They're inverted when comparing to uncertainty estimators (low quality = high uncertainty)

## 2. Uncertainty Estimation Metrics (Correlation Metrics)

These metrics measure **how well your uncertainty estimators correlate with generation quality**. They compare:
- **Uncertainty estimates** (from your estimators like SemanticEntropy, Perplexity, etc.)
- **Generation quality** (from generation metrics above)

**Higher values = better correlation** (uncertainty estimator correctly identifies low-quality generations)

### Available UE Metrics:

#### **PRR - Prediction Rejection Area** (`PredictionRejectionArea`)
- **What it measures**: Area under the Prediction-Rejection curve
- **How it works**: 
  - Sorts samples by uncertainty (lowest to highest)
  - Rejects the most uncertain samples
  - Measures how much quality improves as you reject more samples
- **Range**: 0-1 (higher is better)
- **Interpretation**: If PRR = 0.8, rejecting uncertain samples improves average quality by 80%
- **Use case**: Active learning, selective prediction

#### **RCC - Risk Coverage Curve AUC** (`RiskCoverageCurveAUC`)
- **What it measures**: Area under Risk-Coverage curve
- **How it works**:
  - Similar to PRR but focuses on risk (error rate) vs coverage
  - Measures how risk decreases as you reject uncertain samples
- **Range**: 0-1 (higher is better)
- **Use case**: Selective prediction, quality control

#### **Isotonic PCC** (`IsotonicPCC`)
- **What it measures**: Pearson correlation after isotonic calibration
- **How it works**: Monotonically transforms uncertainty scores to maximize correlation
- **Range**: -1 to 1 (higher is better)
- **Use case**: Calibration assessment

#### **ECE - Expected Calibration Error** (`ECE`)
- **What it measures**: How well-calibrated the uncertainty estimates are
- **How it works**: Compares predicted confidence to actual accuracy in bins
- **Range**: 0-1 (lower is better, but reported as higher is better after inversion)
- **Use case**: Calibration evaluation

#### **Spearman Rank Correlation** (`SpearmanRankCorrelation`)
- **What it measures**: Rank correlation between uncertainty and quality
- **How it works**: Non-parametric correlation (doesn't assume linear relationship)
- **Range**: -1 to 1 (higher is better)
- **Use case**: General correlation assessment

#### **Kendall Tau** (`KendallTauCorrelation`)
- **What it measures**: Rank correlation using concordant/discordant pairs
- **How it works**: Similar to Spearman but uses different method
- **Range**: -1 to 1 (higher is better)
- **Use case**: Rank correlation assessment

#### **RPP - Reversed Pairs Proportion** (`ReversedPairsProportion`)
- **What it measures**: Proportion of pairs where uncertainty ranking is reversed
- **How it works**: Counts pairs where low uncertainty sample has worse quality than high uncertainty sample
- **Range**: 0-1 (lower is better)
- **Use case**: Ranking quality assessment

#### **ROC AUC** (`ROCAUC`)
- **What it measures**: Area under ROC curve for binary classification (good vs bad quality)
- **How it works**: Treats quality as binary (above/below threshold) and measures classification performance
- **Range**: 0-1 (higher is better, 0.5 = random)
- **Use case**: Binary quality prediction

#### **PR AUC** (`PRAUC`)
- **What it measures**: Area under Precision-Recall curve
- **How it works**: Similar to ROC but uses precision-recall instead
- **Range**: 0-1 (higher is better)
- **Use case**: Binary quality prediction (especially for imbalanced data)

## How Metrics Are Computed

For each combination of:
- **Uncertainty Estimator** (e.g., SemanticEntropy, Perplexity)
- **Generation Metric** (e.g., ROUGE-L, BLEU)
- **UE Metric** (e.g., PRR, RCC)

The system computes:
```
correlation = UE_Metric(uncertainty_scores, generation_quality_scores)
```

### Example:
- **Estimator**: SemanticEntropy
- **Generation Metric**: ROUGE-L
- **UE Metric**: PRR
- **Result**: How well SemanticEntropy correlates with ROUGE-L quality (via PRR)

## Understanding Your Results

When you load results:
```python
man = UEManager.load('path/to/ue_manager_seed1')

# Generation metrics: quality of each sample
rouge_scores = man.gen_metrics[('sequence', 'RougeMetric_rougeL')]
# Higher = better quality

# Uncertainty estimates: uncertainty for each sample
semantic_entropy = man.estimations[('sequence', 'SemanticEntropy')]
# Higher = more uncertain

# Correlation metrics: how well uncertainty correlates with quality
prr_score = man.metrics[('sequence', 'SemanticEntropy', 'RougeMetric_rougeL', 'PRR')]
# Higher = better correlation (uncertainty correctly identifies low-quality samples)
```

## Which Metrics Matter?

### For Question Answering (your CoQA task):
- **Generation Metrics**: ROUGE-L, BLEU, Accuracy (exact match)
- **UE Metrics**: PRR, RCC (most important for selective prediction)

### For Summarization:
- **Generation Metrics**: ROUGE-1/2/L, BARTScore, AlignScore
- **UE Metrics**: PRR, RCC, Spearman

### For Translation:
- **Generation Metrics**: BLEU, COMET
- **UE Metrics**: PRR, RCC

## Key Takeaways

1. **Generation Metrics** = "How good is the model's output?"
2. **Uncertainty Estimates** = "How uncertain is the model about this output?"
3. **Correlation Metrics** = "Does high uncertainty correctly predict low quality?"

The goal is to have **high correlation metrics** - meaning your uncertainty estimators successfully identify when the model is producing low-quality outputs.

