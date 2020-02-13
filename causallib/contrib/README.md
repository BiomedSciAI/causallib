# Module `causallib.contrib`
This module currently includes additional causal methods contributed to the package 
by causal inference researchers other than `causallib`'s core developers.

The causal models in this module can be slightly more novel then in the ones in `estimation` module. 
However, they should largely adhere to `causallib` API 
(e.g., `IndividualOutcomeEstimator` or `WeightEstimator`).
Since code here is more experimental, 
models might also require additional (and less trivial) package dependencies, 
or have less test coverage.  
Well-integrated models could be transferred into the main `estimation` module in the future.

## Contributed Methods
Currently contributed methods are:

1. Adversarial Balancing: implementing the algorithm described in 
   [Adversarial Balancing for Causal Inference](https://arxiv.org/abs/1810.07406).
   ```python
   from causallib.contrib.adversarial_balancing import AdversarialBalancing

## Dependencies
Each model might have slightly different requirements.  
Refer to the documentation of each model for the additional packages it requires.
  
Requirements for `contrib` models will be concentrated in `contrib/requirements.txt` and should be
automatically installed using the extra-requirements `contrib` flag:  
```shell script
pip install causallib[contrib]
```   
