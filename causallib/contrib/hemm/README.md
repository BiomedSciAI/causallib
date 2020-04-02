# Module `causallib.contrib.hemm`

Implementation of the heterogeneous effect mixture model (HEMM) presented in the [_Interpretable Subgroup Discovery in Treatment Effect Estimation with Application to Opioid Prescribing Guidelines_](https://arxiv.org/abs/1905.03297) paper.

HEMM is used for discovering subgroups with enhanced and diminished treatment effects in a potential outcomes causal inference framework, using sparsity to enhance interpretability. The HEMM’s outcome model is extended to include neural networks to better adjust for confounding and develop a joint inference procedure for the overall graphical model and neural networks. The model has two parts:

  1. The subgroup discovery component. 
  2. The outcome prediction from the subgroup assignment and the interaction with confounders through an MLP.

The model can be initialized with any of the following outcome models:
  * **Balanced Net**: A torch.model class that is used as a component of the HEMM module to determine the outcome as a function of confounders. The balanced net consists of two different neural networks for the two potential outcomes (under treatment and under control).
  * **MLP model**: An MLP with an ELU activation. This allows for a single neural network to have two heads, one for each of the potential outcomes.
  * **Linear model**: Linear model with two separate linear functions of the input covariates.

The balanced net outcome model relies on utility functions that are to be used with the balanced net outcome model based on [_Estimating individual treatment effect: generalization bounds and algorithms_](https://arxiv.org/abs/1606.03976), Shalit et al., ICML (2017). The utility functions mainly consist of IPM metrics to calculate the imbalance between the control and treated population.
