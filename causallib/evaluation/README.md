# Module `causallib.evaluation`

This submodule allows evaluating the performance of the estimation models defined
in `causallib.estmation`.

The intended usage is to use `evaluate` from `causalib.evaluation` to generate `EvaluationResults` objects. 
If the cross-validation parameter `cv` is not supplied, a simple evaluation without cross-validation will 
be performed. And an object will be returned that can generate various plots, accessible by name (see the docs)
or all at once via `plot_all()`.
The object also includes the model's predictions, evaluated metrics, the fitted models as `models`
and a copy of the original data as (`X`, `a`, and `y`).

If the `cv` parameter is set to `"auto"`, `evaluate` generates a k-fold cross-validation with train and validation
phases, refitting the model `k` times, with `k=5`. Other options are also supported
for customizing cross-validation, see the docs.
The `EvaluationResults` will also contain a list of train/test split indices used by cross-validation in `cv`.


## Example: Inverse probability weighting

An IPW method with logistic regression can be evaluated
in cross-validation using

```Python
from sklearn.linear_model import LogisticRegression
from causallib.estimation import IPW
from causallib.datasets.data_loader import fetch_smoking_weight
from causallib.evaluation import evaluate

data = fetch_smoking_weight()

model = LogisticRegression()
ipw = IPW(learner=model)
ipw.fit(data.X, data.a, data.y)
res = evaluate(ipw, data.X, data.a, data.y, cv="auto")

res.plot_all()
```

This will train the models and create evaluation plots
showing the performance on both the training and validation data.

```python
print(res.all_plot_names)
# {'weight_distribution', 'pr_curve', 'covariate_balance_love', 'roc_curve', 'calibration', 'covariate_balance_slope'}
res.plot_covariate_balance(kind="love", phase="valid")
res.plot_weight_distribution()
res.plot_roc_curve()
res.plot_calibration_curve()
```

## Submodule structure

*This section is intended for future contributors and those seeking to customize the evaluation logic.*

The `evaluate` function is defined in `evaluator.py`. To generate predictions
it instantiates a `Predictor` object as defined in `predictor.py`. This handles
refitting and generating the necessary predictions for the different models.
The predictions objects are defined in `predictions.py`.
Metrics are defined in `metrics.py`. These are simple functions and do not depend
on the structure of the objects.
The metrics are applied to the individual predictions via the scoring functions
defined in `scoring.py`.
The results of the predictors and scorers across multiple phases and folds are
combined in the `EvaluationResults` object which is defined in `results.py`.

### evaluation.plots submodule structure

In order to generate the correct plots from the `EvaluationResults` objects, we
need `PlotDataExtractor` objects. The responsibility of these objects is to extract
the correct data for a given plot from `EvaluationResults`, and they are defined
in `plots/data_extractors.py`.
Enabling plotting as member functions for `EvaluationResults` objects is accomplished
using the plotter mixins, which are defined in `plots/mixins.py`.
When an `EvaluationResults` object is produced by evaluate, the `EvaluationResults.make`
factory ensures that it has the correct extractors and plotting mixins.

Finally, `plots/curve_data_makers.py` contains a number of methods for aggregating and
combining data to produce curves for ROC, PR and calibration plots.
And `plots/plots.py` contains the individual plotting functions.

## How to add a new plot

If there is a model evaluation plot that you would like to add to the codebase,
you must first determine for what models it would be relevant. For example,
a confusion matrix makes sense for a classification task but not for continuous
outcome prediction, or sample weight calculation.

Currently, the types of models are

* Individual outcome predictions (continuous outcome)
* Individual outcome predictions (binary outcome)
* Sample weight predictions
* Propensity predictions

Propensity predictions combine binary individual outcome predictions (because
"is treated" is a binary feature) with sample weight predictions. Something like
a confusion matrix would make sense for binary outcome predictions and for propensity
predictions, but not for the other categories. In that sense it would behave like
the ROC curve, and PR curve which are already implemented.

Assuming you want to add a new plot, you would add the basic plotting
function to `plots/plots.py`. Then you would add a case to the relevant extractors'
`get_data_for_plot` members to extract the data for the plot, based on its name, in `plots/data_extractors.py` . You would also add the name as an available plot in the relevant
`frozenset` and in the `lookup_name` function, both in `plots/plots.py`. At this point, the plot should be drawn automatically when you run `plot_all` on the relevant `EvaluationResults` object.
To expose the plot as a member `plot_my_new_plot`, you must add it to the correct mixin in
`plots/mixins.py`.
