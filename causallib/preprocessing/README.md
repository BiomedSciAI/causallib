# Module `preprocessing`
This module provides several useful filters and transformers to augment
the ones provided by scikit-learn.
 
Specifically, the various filters remove features for the following criteria:
- Features that are almost constant (not by variance but by actual value).
- Features that are highly correlated with other features.
- Features that have a low variance (can deal with NaN values).
- Features that are mostly NaN.
- Features that are highly associated with the outcome (not just correlation)

Various transformers are provided:
- A standard scaler that deals with Nan values.
- A min/max scaler.

A transformer that accepts numpy arrays and turns them into pandas will be added soon. 

These filters and transformers can be used as part of a scikit-learn pipeline.

### Example:
This example combines a scikit-learn filter with a causallib scaler.
The pipeline scales the data, then removes covariates with low variance, 
and then applies IPW with logistic regression. 

```Python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from causallib.estimation import IPW
from causallib.datasets import load_nhefs
from causallib.preprocessing.transformers import MinMaxScaler

pipeline = make_pipeline(MinMaxScaler(), VarianceThreshold(0.1), LogisticRegression())
data = load_nhefs()
ipw = IPW(pipeline)
ipw.fit(data.X, data.a)
ipw.estimate_population_outcome(data.X, data.a, data.y)
```