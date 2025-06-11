# LM-Polygraph Normalization Methods

LM-Polygraph implements several uncertainty normalization methods to convert raw uncertainty scores into more interpretable confidence values bounded between 0 and 1. Here are the key normalization approaches:

### MinMax Normalization (MinMaxNormalizer in `minmax.py`)

- Takes raw uncertainty scores and linearly scales them to [0,1] range.
- Flips the sign since uncertainty scores should be negatively correlated with confidence.
- Uses scikit-learn's `MinMaxScaler` internally.
- Simple but doesn't maintain a direct connection to output quality.

### Quantile Normalization (QuantileNormalizer in `quantile.py`)

- Transforms uncertainty scores into their corresponding percentile ranks.
- Uses empirical CDF to map scores to [0,1] range.
- Provides uniformly distributed confidence scores.
- May lose some granularity of original uncertainty estimates.

## Performance-Calibrated Confidence (PCC) Methods

### Binned PCC (BinnedPCCNormalizer in `binned_pcc.py`)

- Splits calibration data into bins based on uncertainty values.
- Each bin has approximately an equal number of samples.
- Confidence score is the mean output quality of samples in the corresponding bin.
- Provides an interpretable connection between confidence and expected quality.
- Drawback: Can change ordering of samples compared to raw uncertainty.

### Isotonic PCC (IsotonicPCCNormalizer in `isotonic_pcc.py`)

- Uses Centered Isotonic Regression (CIR) to fit a monotonic relationship.
- Maps uncertainty scores to output quality while preserving ordering.
- Enforces monotonicity constraint to maintain uncertainty ranking.
- More robust than the binned approach while maintaining interpretability.
- Implementation based on CIR algorithm from Oron & Flournoy (2017).

## Common Interface: `BaseUENormalizer`

All normalizers follow a common interface defined in `BaseUENormalizer`:

- `fit()`: Learns normalization parameters from calibration data.
- `transform()`: Applies normalization to new uncertainty scores.
- `dumps()/loads()`: Serialization support for fitted normalizers.

## Key Benefits of PCC Methods

- Direct connection to output quality metrics.
- Bounded interpretable range [0,1].
- Maintained correlation with generation quality.
- Easy to explain meaning to end users.

## Highlight: Isotonic PCC

The Isotonic PCC approach provides the best balance between:

- Maintaining the original uncertainty ranking.
- Providing interpretable confidence scores.
- Establishing a clear connection to expected output quality.

When using normalized scores, users can interpret them as estimates of relative output quality, making them more useful for downstream applications and human understanding.