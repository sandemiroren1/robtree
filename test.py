import unittest
from numbers import Number
import subprocess
import numpy as np
from roct.maxsat import SATOptimalRobustTree
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
import exhaustivesearch.RobTree as model

class TestStringMethods(unittest.TestCase):

    
    def run_roct_test(self, n, m, depth, epsilon=0.1):
        X = np.random.rand(n, m)*10
        Y = np.random.randint(0, 2, size=n)
        X = MinMaxScaler().fit_transform(X)
        X = np.round(X,6)
        attack_model = [epsilon] * m


        tree = (model.OptimalRobustTreeSearch(attack_model=attack_model,s=300))
        tree2 = (OptimalRobustTree(attack_model=attack_model, max_depth=depth))

        print(f"Running test with n={n}, m={m}, depth={depth}")
        time_now = time.time()
        tree.fit(X, Y)
        print("Search time: ", time.time() - time_now)
        
        time_now = time.time()
        tree2.fit(X, Y)
        print("ROCT time: ", time.time() - time_now)
        

        tree = Model.from_groot(tree)
        tree2 = Model.from_groot(tree2) 
        accuracy_search = tree.adversarial_accuracy(X, Y, attack="tree", epsilon=epsilon)
        accuracy_roct = tree2.adversarial_accuracy(X, Y, attack="tree", epsilon=epsilon)
        
        if(accuracy_roct != accuracy_search):
            print("Accuracy: search=", accuracy_search, ", roct=", accuracy_roct)
            np.set_printoptions(precision=20, suppress=False)
            x_str = "X = [\n" + "\n".join(
            "  [" + ", ".join(f"{x:.20f}" for x in row) + "]," for row in X
                ) + "\n]"
            y_str = "Y = [" + ", ".join(str(y) for y in Y) + "]"
            print(x_str)
            print(y_str)
            #write them to file
            with open("bad-case.txt", "w") as f:
                f.write(x_str)
                f.write(y_str)
        self.assertEqual(accuracy_search, accuracy_roct)
    
    def hard_coded(self, X,Y,epsilon, accuracy):
        X = MinMaxScaler().fit_transform(X)
        X = np.round(X,6)
        attack_model = [epsilon] * X.shape[1]

        groot_tree = GrootTreeClassifier(
        max_depth=2, attack_model=attack_model, min_samples_split=2, random_state=1
        )
        groot_tree.fit(X, Y)
        warm_value = len(Y)-np.ceil(len(Y)*Model.from_groot(groot_tree).adversarial_accuracy(X, Y, attack = "tree", epsilon=epsilon))+1
        tree = model.OptimalRobustTreeSearch(attack_model=attack_model,s=3,warm_start_value=warm_value)

        tree.fit(X, Y)
        tree = Model.from_groot(tree)
        accuracy_search = tree.adversarial_accuracy(X, Y, attack="tree", epsilon=epsilon)
        self.assertEqual(accuracy_search,accuracy)
    # Individual test cases
    def test_haberman_02(self):
        X = np.load("data/X_train_haberman.npy")
        Y = np.load("data/Y_train_haberman.npy")
        self.hard_coded(X,Y,0.02,0.7663934426229508)
    def test_haberman_03(self):
        X = np.load("data/X_train_haberman.npy")
        Y = np.load("data/Y_train_haberman.npy")
        self.hard_coded(X,Y,0.03,0.7581967213114754)

    def test_haberman_05(self):
        X = np.load("data/X_train_haberman.npy")
        Y = np.load("data/Y_train_haberman.npy")
        self.hard_coded(X,Y,0.05,0.7377049180327869)

    def test_breast_cancer_28(self):
        X = np.load("data/X_train_breast-cancer.npy")
        Y = np.load("data/Y_train_breast-cancer.npy")
        self.hard_coded(X,Y,0.28,0.8681318681318682)
    def test_breast_cancer_39(self):
        X = np.load("data/X_train_breast-cancer.npy")
        Y = np.load("data/Y_train_breast-cancer.npy")
        self.hard_coded(X,Y,0.39,0.8095238095238095)
    def test_breast_cancer_45(self):
        X = np.load("data/X_train_breast-cancer.npy")
        Y = np.load("data/Y_train_breast-cancer.npy")
        self.hard_coded(X,Y,0.45,0.7509157509157509)
    def test_blood_transfusion_01(self):
        X = np.load("data/X_train_blood-transfusion.npy")
        Y = np.load("data/Y_train_blood-transfusion.npy")
        self.hard_coded(X,Y,0.01,0.7859531772575251)
    def test_blood_transfusion_02(self):
        X = np.load("data/X_train_blood-transfusion.npy")
        Y = np.load("data/Y_train_blood-transfusion.npy")
        self.hard_coded(X,Y,0.02,0.774247491638796)
    def test_blood_transfusion_03(self):
        X = np.load("data/X_train_blood-transfusion.npy")
        Y = np.load("data/Y_train_blood-transfusion.npy")
        self.hard_coded(X,Y,0.03,0.7725752508361204)
    
        self.hard_coded(X,Y,0.11,0.6545123062898814)
    #def test_wine_002(self):
       # X = np.load("data/X_train_wine.npy")
      #  Y = np.load("data/Y_train_wine.npy")
     #   self.hard_coded(X,Y,0.02,0.6790456032326342)
    #def test_wine_003(self):
        #X = np.load("data/X_train_wine.npy")
        #Y = np.load("data/Y_train_wine.npy")
        #self.hard_coded(X,Y,0.03,0.6584568020011545)
    #def test_wine_004(self):
     #   X = np.load("data/X_train_wine.npy")
      #  Y = np.load("data/Y_train_wine.npy")
       # self.hard_coded(X,Y,0.04,0.6486434481431596)
    def test_diabetes_005(self):
        X = np.load("data/X_train_diabetes.npy")
        Y = np.load("data/Y_train_diabetes.npy")
        self.hard_coded(X,Y,0.05,0.6856677524429967)
    def test_diabetes_007(self):
        X = np.load("data/X_train_diabetes.npy")
        Y = np.load("data/Y_train_diabetes.npy")
        self.hard_coded(X,Y,0.07,0.6677524429967426)
    def test_diabetes_009(self):
        X = np.load("data/X_train_diabetes.npy")
        Y = np.load("data/Y_train_diabetes.npy")
        self.hard_coded(X,Y,0.09,0.6612377850162866)
    def test_cylinder_bands_023(self):
        X = np.load("data/X_train_cylinder-bands.npy")
        Y = np.load("data/Y_train_cylinder-bands.npy")
        self.hard_coded(X,Y,0.23,0.7149321266968326)
    def test_cylinder_bands_028(self):
        X = np.load("data/X_train_cylinder-bands.npy")
        Y = np.load("data/Y_train_cylinder-bands.npy")
        self.hard_coded(X,Y,0.28,0.7013574660633484)
    def test_cylinder_bands_045(self):
        X = np.load("data/X_train_cylinder-bands.npy")
        Y = np.load("data/Y_train_cylinder-bands.npy")
        self.hard_coded(X,Y,0.45,0.7013574660633484)
    def test_random_res(self):
        for _ in range(100):
            self.run_roct_test(6,2,2,0.1)
    def test_random_res_aggresive_adversary(self):
        
        for _ in range(100):
            self.run_roct_test(10,2,2,0.5)
    def test_random_big(self):
        for _ in range(30):
            self.run_roct_test(10,4,2,0.1)

    

if __name__ == '__main__':
    unittest.main()