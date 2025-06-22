import unittest
from numbers import Number
import subprocess
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import exhaustivesearch.RobTree as model
import time
from roct.milp import OptimalRobustTree
from groot.adversary import DecisionTreeAdversary
from groot.adversary import DecisionTreeAdversary
from groot.datasets import epsilon_attacker
from groot.model import GrootTreeClassifier
from groot.treant import RobustDecisionTree
from groot.visualization import plot_adversary
import itertools
from groot.toolbox import Model
print("starting...\n")
from roct.maxsat import SATOptimalRobustTree




def test_hard_coded(X, Y, epsilon):
        X = MinMaxScaler().fit_transform(X)
        X = np.round(X,6)
        attack_model = [epsilon] * X.shape[1]

        groot_tree = GrootTreeClassifier(
        max_depth=2, attack_model=attack_model, min_samples_split=2, random_state=1
        )
        groot_tree.fit(X, Y)
        
        tree = model.OptimalRobustTreeSearch(attack_model=attack_model,s=3000,time_limit=1<<30,warm_start_value=1<<30,record_progress=True)
        time_now = time.time()
        tree.fit(X, Y)
        print("Search time: ", time.time() - time_now)
        tree = Model.from_groot(tree)
        accuracy_search = tree.adversarial_accuracy(X, Y, attack="tree", epsilon=epsilon)

        print("search accuracy",accuracy_search) 

        tree2 = SATOptimalRobustTree(
        max_depth=2, attack_model=attack_model, lsu=True, time_limit=600
        )
        time_now = time.time()
        tree2.fit(X, Y)
        tree2 = Model.from_groot(tree2)
                
        print("ROCT time: ", time.time() - time_now)
        accuracy_roct = tree2.adversarial_accuracy(X, Y, attack="tree", epsilon=epsilon)
        print("roct accuracy",accuracy_roct) 

# You can uncomment the following lines to test with different datasets.

#X =   np.load("data/X_train_breast-cancer.npy")
#X = np.load("data/X_train_blood-transfusion.npy")
#X =   np.load("data/X_train_cylinder-bands.npy")
#X = np.load("data/X_train_banknote-authentication.npy")
#X = np.load("data/X_train_haberman.npy")
X = np.load("data/X_train_ionosphere.npy")
#X = np.load("data/X_train_diabetes.npy")
#X = np.load("data/X_train_wine.npy")

#Y = np.load("data/Y_train_diabetes.npy")

#Y =   np.load("data/Y_train_breast-cancer.npy")
#Y = np.load("data/Y_train_blood-transfusion.npy")

#Y =   np.load("data/Y_train_cylinder-bands.npy")
#Y =   np.load("data/Y_train_banknote-authentication.npy")
#Y = np.load("data/Y_train_haberman.npy")
Y = np.load("data/Y_train_ionosphere.npy")
#Y = np.load("data/Y_train_wine.npy")


epsilon = 0.1


test_hard_coded(X,Y,epsilon)




