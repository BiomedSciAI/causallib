# Package `causallib`
A package for estimating causal effect and counterfactual outcomes from observational data.

`casuallib` provide various causal inference methods with a distinct paradigm:
 * Every causal model has some machine learning model at its core. 
   This allows to mix & match causal models with powerful machine learning tools, 
   simply by plugging them into the causal model.
 * Inspired by the scikit-learn design, once trained, causal models can be 
   applied onto out-of-bag samples.

`causallib` also provide performance evaluation scheme of the causal model 
by evaluating the machine learning core model in a causal inference context.

Accompanying datasets are also available, both real and simulated ones.   
The various modules and folders provide the specific usage for each part.

## Structure
The package is comprised of several modules, 
each providing a different functionality 
that is related to the causal inference models. 

### `estimation`
This module includes the estimator classes, 
where multiple popular estimators are implemented. 
Specifically, This includes
- Inverse probability weighting (IPW).
- Standardization.
- 3 versions of doubly-robust methods.

Each of these methods receives one or more machine learning models that 
can be trained (fit), and then used to estimate (predict) the relevant outcome
of interest.

### `evaluation`
This module provides the classes to evaluate the performance of methods 
defined in the estimation module.
Evaluations are tailored to the type of method that is used. 
For example, weight estimators such as IPW can be evaluated for how well
they remove bias from the data, 
while outcome models can be evaluated for their precision.

### `preprocessing`
This module provides several enhancements to the filters and transformers
provided by scikit-learn.
These can be used within a pipeline framework together with the models.

### `datasets`
Several datasets are provided within the package in the `datasets` module:  
* NHEFS study data on the effect of smoking cessation on weight gain.
  Adapted from [Hernán and Robins' Causal Inference Book](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)
* A handful of simulation sets from the [2016 Atlantic Causal Inference 
  Conference (ACIC) data challenge](https://jenniferhill7.wixsite.com/acic-2016/competition). 
* Simulation module allows creating simulated data based on a causal graph
  depicting the connection between covariates, treatment assignment and outcomes.

### Additional folders
Several additional folders exist under the package and hold several
internal utilities.
They should only be used as part of development.
This folders include `analysis`, `simulation`, `utils`, and `tests`. 
 
## Usage
The examples folder contains several notebooks exemplifying the use of the 
package.
 