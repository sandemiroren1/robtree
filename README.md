# RobTree: A Scalable Search Algorithm for Optimal Robust Decision Trees

This folder contains the scripts and code needed to reproduce the results of 'The Search for Optimal Robust Decision Trees'. Much of our code extends from the of 'ROCT' at https://github.com/tudelft-cda-lab/ROCT which extends from 'GROOT' at https://github.com/tudelft-cda-lab/GROOT. Lets see if the chain will keep going!

To run our experiments you need to clone our repository. Moreover, you need to download the required packages using `pip install -r requirements.txt`.

The code needs a new version of python, at least 3.7.

## Simple example
Below is a small example for running RobTree.

```python3
from numbers import Number
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import exhaustivesearch.RobTree as model
from groot.adversary import DecisionTreeAdversary
from groot.adversary import DecisionTreeAdversary
from groot.datasets import epsilon_attacker
from groot.treant import RobustDecisionTree
from groot.toolbox import Model

# Load the dataset
X, y = make_moons(noise=0.3, random_state=0)
X_test, y_test = make_moons(noise=0.3, random_state=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Define the attacker's capabilities (L-inf norm radius 0.3)
epsilon = 0.1
attack_model = [epsilon, epsilon]

names = ("MILP", "Binary MILP", "MaxSAT")
tree = model.OptimalRobustTreeSearch(attack_model=attack_model,s=3)

tree.fit(X, y)

# Determine the accuracy and adversarial accuracy against attackers
accuracy = tree.score(X_test, y_test)
model = Model.from_groot(tree)
adversarial_accuracy = model.adversarial_accuracy(X_test, y_test, attack="tree", epsilon=epsilon)

print(name)
print("Accuracy:", accuracy)
print("Adversarial Accuracy:", adversarial_accuracy)
```
## Tests

Tests can be found in `test.py`. It tests the model on pre-known values and randomly generated points.
## Reproducing results
Our figures and fitted trees can be accessed under the `out/` directory, but the results can also be generated from scratch. Fitting all trees and running all experiments can take many days depending on how many parallel cores are available.
### Compiling the solver

The solver code can be found in `exhaustivesearch/solvernewalg.cpp`. This is the variation in the paper. On g++ we recommend compiling with `g++ -g -o solvernewalg.exe -O3 -Ofast  exhaustivesearch/solvernewalgorithm.cpp`. For clang, we recommend `clang++ exhaustivesearch/solvernewalgorithm.cpp -O3 -flto=thin -fuse-ld=lld -std=c++20  -g -o solvernewalg.exe`. The code must be compiled before running the algorithm through python.

### Downloading the datasets
Running the script `download_datasets.py` will download all required datasets from openML into the `data/` directory (which has already been done).

### Main results
The main script for fitting and scoring trees is `run.py`, which can be accessed by a command line interface. To create the jobs to use for your experiments, you can use the `process_results.py` file. The experiments can be then run with (on four cores):
```
parallel -j 4 :::: all_jobs_depth2.txt
```
The resulting trees will generate under `out/results/`. To generate figures and tables please run `python process_results.py` after the parallel process has finished.

### Solver progress over time
The script for plotting solver progress over time is `performance_over_time.py`. This script runs each early-stoppable solver one after the other with trees of depth 2 on one epsilon setting per dataset. This script does not run in parallel so it can take many hours to run.

