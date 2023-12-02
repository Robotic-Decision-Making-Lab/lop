# Learning Objective functions from Preferences (LOP)
[![Python Testing](https://github.com/Robotic-Decision-Making-Lab/lop/actions/workflows/python-test.yml/badge.svg)](https://github.com/Robotic-Decision-Making-Lab/lop/actions/workflows/python-test.yml)


Learn Objective Function From Preferences. An implentation of learning objective function for robotics from user preferences. Includes software to learn both functions with both linear and non-linear functions using Gaussian Processes.


## Dependencies

numpy

matplotlib>=

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
Or using the automatically generated [documentation](http://todo).





## Usage

```
import lop

# TODO lol, good luck

```

### Pairwise preferences

Encoding a single pairwise preference (example1 > example2) for the GP.
These are encoded as (preference, index1, index2).
Where, preference/dk is -1 indicates f(index1) > f(index2), 1 indicates f(index1) < f(index2).
This can be automatically encoded using lop.dk(value1, value2).

