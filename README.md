# Learning Objective functions from Preferences (LOP)
[![Python Testing](https://github.com/Robotic-Decision-Making-Lab/lop/actions/workflows/python-test.yml/badge.svg)](https://github.com/Robotic-Decision-Making-Lab/lop/actions/workflows/python-test.yml)


Learn Objective Function From Preferences. An implentation of learning objective function for robotics from user preferences. Includes software to learn both functions with both linear and non-linear functions using Gaussian Processes.


## Dependencies

See setup.py

## Installation
From the root directory of lop
```
pip install -e . --user
```

## Documentation

Doxygen documntation is used for the library. This allows the creation of html files for the docs.
Generate the documentation locally using:
```
sudo apt-get install doxygen
cd doc
doxygen Doxyfile
```
Or using the automatically generated [documentation](https://robotic-decision-making-lab.github.io/lop/index.html).





## Usage

```
import lop


# setup the mixed-type query selection active learning algorithm
abs_comp = lop.AbsAcquisition(M=M, alignment_f='spearman')
pair = lop.AcquisitionSelection(M=M, alignment_f='spearman')
al = lop.MixedComparision(pairwise_l=pair, abs_l=lop.UCBLearner(), abs_comp=abs_comp)

# setup the preference GP model
# check examples and documentation for more detailed usage of this model
model = lop.PreferenceGP(lop.RBF_kern(1.0, 1.0), active_learner=al)


# learn the preferences
for i in range(...):
    # generate possible canidate plans

    # find the query (2 plans or 1 if selects a rating plan)
    query = model.select(canidiates, 2)

    x = canidiates[query]

    if len(query) == 1:
        rating = # human specified rating between 0 and 1

        model.add(x, rating, type='abs)

    else:
        pairs = # human specified preference
        
        model.add(x, pairs)


```

### Examples

In the examples scripts there are several different examples to see different aspects of the preference learning. 

For a basic usage of the Preference based Gaussian Process, see 
```python3 examples/models/pref_gp.py```

For active learning with preferences see 
```python3 examples/active_learning/pref_gp_ucb_learner.py  --selector ACQ_SPEAR```
 which has several different options for selecting different active learning.

For mixed-type query selection see 
```python3 examples/active_learning/switch_gp_learner.py``` 
which also has different options for different active learning.

For hyperparameter optimization [SUPER FINICKY AND ONLY KIND OF WORKS] examples/hyperparameters/hyperparameter_gp.py

### Pairwise preferences

Encoding a single pairwise preference (example1 > example2) for the GP.
These are encoded as ```(preference, index1, index2)```.
Where, preference/dk is -1 indicates f(index1) > f(index2), 1 indicates f(index1) < f(index2).
This can be automatically encoded using lop.dk(value1, value2).

