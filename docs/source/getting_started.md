# Getting Started

Welcome to causallib! This guide will help you get started with causal inference in Python.

## Installation

Install causallib using pip:

```bash
pip install causallib
```

We recommend to always use a virtual environment for your projects. 
If this concept is unfamiliar to you, please make sure to check out [uv](https://docs.astral.sh/uv/pip/environments/) or [(mini)conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). 

If you want to take advantage of some of the more advanced methods in the `contrib` module, you'll need to install the `contrib` extra:

```bash
pip install causallib[contrib]
```

For development installation with all dependencies:

```bash
git clone https://github.com/BiomedSciAI/causallib.git
cd causallib
pip install -e ".[dev,docs]"

# Or the pinned versions with:
pip install -r requirements.txt -r docs/requirements.txt
```

## Quick Start

Here's a simple example using Inverse Propensity Weighting (IPW) to estimate causal effects:

```python
from causallib.estimation import IPW
from causallib.datasets import load_nhefs
from sklearn.linear_model import LogisticRegression

# Load example data
data = load_nhefs()

# Create and fit IPW model
ipw = IPW(LogisticRegression())
ipw.fit(data.X, data.a)

# Estimate population outcomes
outcomes = ipw.estimate_population_outcome(data.X, data.a, data.y)
print(f"Outcome under control: {outcomes[0]:.2f}")
print(f"Outcome under treatment: {outcomes[1]:.2f}")

# Estimate treatment effect
effect = ipw.estimate_effect(outcomes[1], outcomes[0])
print(f"Average Treatment Effect: {effect['diff']:.2f}")
```

## Core Concepts

### Modular flexibility
causallib provides multiple methods (causal estimators) for estimating causal effects,
each can usually take advantage of arbitrary machine learning estimators under the hood.

This strategy allows modular flexibility by mixing and matching arbitrary causal and ML estimators, as long as they adhere to Scikit-Learn's  `fit` and `predict` API.<br>
It is also makes causallib efficient, as it is often a relatively thin wrapper around statistical estimators who do most of the heavy lifting of compute.
So causallib often works even when the underlying estimators are uniquely tailored models (like XGBoost or Spark).


### Available causal estimators
- **IPW (Inverse Propensity Weighting)**: Reweights samples by inverse probability of treatment
- **Standardization**: Direct outcome modeling (S-Learner, T-Learner)
- **Doubly Robust Methods**: Combines propensity and outcome models (AIPW, TMLE...)
- **Meta-Learners**: More elaborate ways to utilize flexible machine learning models (R-Learner, X-Learner)
- **Matching**: Finds similar treated and control units
- **Survival models**: For time-to-event data

### Data Structure

causallib expects:
- **X**: Covariates/features (pandas DataFrame)
- **a**: Treatment assignment (pandas Series)
- **y**: Outcome variable (pandas Series)
- **t**: Optional time variable for survival analysis (pandas Series)

### Model Evaluation

Evaluate your causal models using built-in diagnostics:

```python
from causallib.evaluation import evaluate

# Evaluate with cross-validation
results = evaluate(ipw, X, a, y, cv="auto")

# Plot diagnostics
results.plot_all()

# Get detailed covariate balance
results.evaluated_metrics.covariate_balance

# Get prediction performance summary
results.evaluated_metrics.prediction_scores
```

## Next Steps

- **[User Guide](user_guide/index.md)**: Learn about different estimation methods
- **[Examples](examples/index.md)**: Explore real-world applications
- **[API Reference](api/index.md)**: Detailed API documentation


## Resources

- **GitHub**: [https://github.com/BiomedSciAI/causallib](https://github.com/BiomedSciAI/causallib)
- **PyPI**: [https://pypi.org/project/causallib/](https://pypi.org/project/causallib/)

## Getting Help

- Check the [Examples](examples/index.md) for common use cases
- Review the [API Reference](api/index.md) for detailed documentation
- Open an issue on [GitHub](https://github.com/BiomedSciAI/causallib/issues) for bugs or feature requests