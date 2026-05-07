# API Reference

Automatically-generated API documentation.

```{eval-rst}
.. currentmodule:: causallib
```

## `estimation`

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~estimation.ipw.IPW
   ~estimation.OverlapWeights
   ~estimation.matching.Matching
   ~estimation.PropensityMatching
   ~estimation.standardization.Standardization
   ~estimation.StratifiedStandardization
   ~estimation.tmle.TMLE
   ~estimation.AIPW
   ~estimation.PropensityFeatureStandardization
   ~estimation.WeightedStandardization
   ~estimation.rlearner.RLearner
   ~estimation.xlearner.XLearner
   ~estimation.MarginalOutcomeEstimator
```

## `survival`

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~survival.WeightedSurvival
   ~survival.StandardizedSurvival
   ~survival.WeightedStandardizedSurvival
   ~survival.MarginalSurvival
   ~survival.RegressionCurveFitter
   ~survival.UnivariateCurveFitter
```

## `preprocessing`

### `transformers`

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~preprocessing.transformers.PropensityTransformer
   ~preprocessing.transformers.MatchingTransformer
```

### `confounder_selection`

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~preprocessing.confounder_selection.DoubleLASSO
   ~preprocessing.confounder_selection.RecursiveConfounderElimination
```

### `positivity`

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~positivity.trimming.Trimming
   ~positivity.UnivariateBoundingBox
```

## model selection and evaluation

### `metrics` 

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~metrics.get_scorer
   ~metrics.get_scorer_names
```

### `model_selection`

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:


   ~model_selection.causalize_searcher
   ~model_selection.GridSearchCV
   ~model_selection.RandomizedSearchCV
   ~model_selection.TreatmentStratifiedKFold
   ~model_selection.TreatmentOutcomeStratifiedKFold
```

### `evaluate`

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~evaluation.evaluate
   ~evaluation.evaluate_bootstrap
```

## `datasets`

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   datasets.load_nhefs
   datasets.load_nhefs_survival
   datasets.load_acic16
   simulation.CausalSimulator3
```

## `contrib`

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~contrib.adversarial_balancing.AdversarialBalancing
   ~contrib.hemm.hemm.HEMM
   ~contrib.bicause_tree.BICauseTree
   ~contrib.faissknn.FaissNearestNeighbors
```



---

## Index and Search

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`