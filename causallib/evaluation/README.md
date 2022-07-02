# Module `causallib.evaluation`

This method allows evaluating the performance of the estimation models defined
in `causallib.estmation`.
Evaluations can be performed in train-test, cross-validation, or bootstrapping
schemes using the functions `evaluate_simple`, `evaluate_cv`, or
`evaluate_bootstrap`, respectively.

### Example: Inverse probability weighting

An IPW method with logistic regression can be evaluated
in cross-validation using

```Python
from sklearn.linear_model import LogisticRegression
from causallib.estimation import IPW
from causallib.datasets.data_loader import fetch_smoking_weight
from causallib.evaluation import Evaluator, plot_evaluation_results

data = fetch_smoking_weight()

model = LogisticRegression()
ipw = IPW(learner=model)
evaluator = Evaluator(ipw)

res = evaluator.evaluate_cv(data.X, data.a, data.y)

plot_evaluation_results(res, data.X, data.a, data.y)
```

This will train the models and create evaluation plots
showing the performance on both the training and validation data.

To select specific plots only, select a plot from `available_plot_names`

```python
print(res.available_plot_names)
# {'weight_distribution', 'pr_curve', 'covariate_balance_love', 'roc_curve', 'calibration', 'covariate_balance_slope'}
from causallib.evaluation import plot_single_evaluation_result
plot_single_evaluation_result(res, data.X, data.a, data.y, "covariate_balance_love", "valid")
```