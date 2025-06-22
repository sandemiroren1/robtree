from groot.datasets import epsilon_attacker
from groot.model import GrootTreeClassifier
from groot.toolbox import Model
from groot.treant import RobustDecisionTree
from groot.util import convert_numpy
from exhaustivesearch.RobTree import OptimalRobustTreeSearch
from roct.maxsat import SATOptimalRobustTree
from roct.milp import OptimalRobustTree, BinaryOptimalRobustTree
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import numpy as np
output_file = "all_jobs_depth2.txt"  # The updated commands will be saved here

time_limit = 30*60
fixed_depth = 2
algorithms = ["milp-warm","Pure-Search","Pure-Search-warm","lsu-maxsat","groot","RobTree","RobTree-warm"]
datasets = [
    "banknote-authentication",
    "blood-transfusion",
    "breast-cancer",
    "cylinder-bands",
    "diabetes",
    "haberman",
    "ionosphere",
    "wine",
]

epsilons = {
    "banknote-authentication": [0.07,0.09,0.11],
    "blood-transfusion": [0.01,0.02,0.03],
    "breast-cancer": [0.28,0.39,0.45],
    "cylinder-bands": [0.23,0.28,0.45],
    "diabetes": [0.05,0.07,0.09],
    "haberman": [0.02,0.03,0.05],
    "ionosphere": [0.2,0.28,0.36],
    "wine": [0.02,0.03,0.04],
}

timeouts_per_alg = {
    "milp-warm": {"banknote-authentication0.07","banknote-authentication0.09","banknote-authentication0.11",
                  "blood-transfusion0.01","blood-transfusion0.02","blood-transfusion0.03",
                  "cylinder-bands0.23","cylinder-bands0.28","cylinder-bands0.45",
                  "diabetes0.05","diabetes0.07","diabetes0.09",
                  "haberman0.02","haberman0.03","haberman0.05",
                  "ionosphere0.2","ionosphere0.28","ionosphere0.36",
                  "wine0.02","wine0.03","wine0.04"},
    "lsu-maxsat": {
        "banknote-authentication0.11","diabetes0.05","diabetes0.07",
        "wine0.02","wine0.03","wine0.04"},
    "Pure-Search" : {},
    "Pure-Search-warm": {},
    "RobTree" : {},
    "RobTree-warm" : {},

    "groot" : {}
}



with open(output_file, "w") as f_out:
    for alg in algorithms:
        for dataset in datasets:
            for epsilon in epsilons[dataset]:
                # python run.py Pure-Search blood-transfusion -e 0.01 -t 1800 --fix_depth 2
                if f"{dataset}{epsilon}" in timeouts_per_alg[alg]:
                    continue
                f_out.write(f"python run.py {alg} {dataset} -e {epsilon} -t {time_limit} --fix_depth {fixed_depth}\n")

