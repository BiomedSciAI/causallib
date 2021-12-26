# Module `causallib.survival`
This module allows estimating counterfactual outcomes in a setting of right-censored data 
(also known as survival analysis, or time-to-event modeling).
In addition to the standard inputs of `X` - baseline covariates, `a` - treatment assignment and `y` - outcome indicator,
a new variable `t` is introduced, measuring time from the beginning of observation period to an occurrence of event. 
An event may be right-censoring (where `y=0`) or an outcome of interest, or "death" (where `y=1`, 
which is also considered as censoring).  
Each of these methods uses an underlying machine learning model of choice, and can also integrate with the 
[`lifelines`](https://github.com/CamDavidsonPilon/lifelines) survival analysis Python package.

Additional methods will be added incrementally.

## Available Methods
The methods that are currently available are:
1. Weighting: `causallib.survival.WeightedSurvival` - uses `causallib`'s `WeightEstimator` (e.g., `IPW`) to generate weighted pseudo-population for survival analysis.
2. Standardization (parametric g-formula): `causallib.survival.StandardizedSurvival` - fits a parametric hazards model that includes baseline covariates.
3. Weighted Standardization: `causallib.survival.WeightedStandardizedSurvival` - combines the two above-mentioned methods.

### Example: Weighted survival analysis with Inverse Probability Weighting
```python
from sklearn.linear_model import LogisticRegression
from causallib.survival import WeightedSurvival
from causallib.estimation import IPW
from causallib.datasets import load_nhefs_survival

ipw = IPW(learner=LogisticRegression())
weighted_survival_estimator = WeightedSurvival(weight_model=ipw)
X, a, t, y = load_nhefs_survival()

weighted_survival_estimator.fit(X, a)
population_averaged_survival_curves = weighted_survival_estimator.estimate_population_outcome(X, a, t, y)
```

### Example: Standardized survival (parametric g-formula)
```python
from causallib.survival import StandardizedSurvival

standardized_survival = StandardizedSurvival(survival_model=LogisticRegression())
standardized_survival.fit(X, a, t, y)
population_averaged_survival_curves = standardized_survival.estimate_poplatuon_outcome(X, a, t)
individual_survival_curves = standardized_survival.estimate_individual_outcome(X, a, t)
```