# Module `causallib.estimation`
This module allows estimating counterfactual outcomes and effect of treatment
using a variety of common causal inference methods, as detailed below.
Each of these methods can use an underlying machine learning model of choice.
These models must have an interface similar to the one defined by 
scikit-learn.
Namely, they must have `fit()` and `predict()` functions implemented,
and `predict_proba()` implemented for models that predict categorical outcomes.
  
Additional methods will be added incrementally.

## Available Methods
The methods that are currently available are:

1. Inverse probability weighting (with minimal value cutoff): 
`causallib.estimation.IPW`
1. Standardization 
    1. As a single model depending on treatment: 
    `causallib.estimation.Standardization` 
    1. Stratified by treatment value (similar to pooled regression):
    `causallib.estimation.StratifiedStandardization` 
1. Doubly robust methods, as explained 
[here](https://www4.stat.ncsu.edu/~davidian/double.pdf)
    1. Using the weighting as an additional feature:
    `causallib.estimation.DoublyRobustIpFeature`
    1. Using the weighting for training the standardization model:
    `causallib.estimation.DoublyRobustJoffe`
    1. Using the original formula for doubly robust estimation:
    `causallib.estimation.DoublyRobustVanilla` 


### Example: Inverse Probability Weighting (IPW)
An IPW model can be run, for example, using
```Python
from sklearn.linear_model import LogisticRegression
from causallib.estimation import IPW
from causallib.datasets.data_loader import fetch_smoking_weight

model = LogisticRegression()
ipw = IPW(learner=model)
data = fetch_smoking_weight()
ipw.fit(data.X, data.a)
ipw.estimate_population_outcome(data.X, data.a, data.y)
``` 
Note that `model` can be replaced by any machine learning model 
as explained above. 
