# Contribution guidelines

Causallib welcomes community contributions to this repository.  
This file provides the guidelines to contribute to this project.

## Contributions
We welcome a wide range of contributions: 
 - estimation models
 - preprocessors
 - plots
 - improvements to the overall design
 - causal analysis examples using causallib in Jupyter Notebooks 
 - documentation 
 - bug reports
 - bug fixes
 - and more

## Prerequisites
Causallib follows the [Github contribution workflow](https://git-scm.com/book/sv/v2/GitHub-Contributing-to-a-Project):
forking the repository, cloning it, branching out a feature branch, developing,
opening a pull request back to the causallib upstream once you are done,
and performing an iterative review process.  
If your changes require a lot of work, it is better to first make sure they are 
aligned with the plans for the package. 
Therefore, it is recommended that you first open an issue describing 
what changes you think should be made and why.
After a discussion with the core maintainers, we will decide whether the suggestion
is welcomed or not. 
If so, you are encouraged to link you pull request to its corresponding issue.

### Tests
Contribution of new code is best when accompanied by corresponding testing code.
Unittests should be located in the `causallib/tests/` directory and run with `pytest`. 

New bug fixes should, too, be ideally coupled with tests replicating the bug,
ensuring it will not repeat in the future.

### Documentation
New code should also be well documented. 
Causallib uses [Google docstring format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html),
and docstrings should include input and output typing 
(if not [specified in the code](https://docs.python.org/3/library/typing.html)).  
If there are relevant academic papers on which the contribution is based upon, 
please cite it and link to it in the docstring.

### Style
The ultimate goal is to enhance the readability of the code.
Causallib does not currently adhere to any strict style guideline.
It follows the general guidance of PEP8 specifications,
but encourages contributors to diverge from it if they see fit.

Whenever in doubt - follow [the _Black_ code style guide](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html).

### `Contrib` module
The `contrib` module is designated to more state-of-the-art methods that are not
yet well-established but nonetheless may benefit the community.
Ideally, models should still adhere to the causallib's API 
(namely, `IndividualOutcomeEstimator`, `PopulationOutcomeEstimator`, `WeightEstimator`).
This module has its own requirements file and tests.

### Contributor License Agreement
The Causallib developer team works for IBM. 
To accept contributions outside of IBM, 
we need a signed Contributor License Agreement (CLA) 
from you before code contributions can be reviewed and merged.
By signing a contributor license agreement (CLA), 
you're basically just attesting to the fact that 
you are the author of the contribution and that you're freely
contributing it under the terms of the Apache-2.0 license.

When you contribute to the Causallib project with a new pull request,
a bot will evaluate whether you have signed the CLA. If required, the
bot will comment on the pull request, including a link to accept the
agreement. 
You can review the [individual CLA document as a PDF](https://www.apache.org/licenses/icla.pdf).

**Note**:
> If your contribution is part of your employment or your contribution
> is the property of your employer, then you will likely need to sign a
> [corporate CLA](https://www.apache.org/licenses/cla-corporate.txt) too and
> email it to us at <ehudk@ibm.com>.

## Contributors
Ehud Karavani  
Yishai Shimoni  
Michael Danziger  
Lior Ness  
Itay Manes  
Yoav Kan-Tor  
Chirag Nagpal   
Tal Kozlovski  
Liran Szlak  
Onkar Bhardwaj  
Dennis Wei  
