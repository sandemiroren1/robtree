import math
import os
import numpy as np
import subprocess
from groot.adversary import DecisionTreeAdversary
from groot.model import (
    GrootTreeClassifier,
    NumericalNode,
    Node,
    _TREE_LEAF,
    _TREE_UNDEFINED,
)
from numbers import Number

from exhaustivesearch.base import BaseOptimalRobustTree
class OptimalRobustTreeSearch(BaseOptimalRobustTree):
    # Currently only works for depth 2.
    def __init__(
        self,
        attack_model=None,
        time_limit=None,
        record_progress=False,
        verbose=False,
        input_file_name = "input.txt",
        warm_start_value= None,
        s = 3
    ):
        super().__init__(
            max_depth=2,
            attack_model=attack_model,
            time_limit=time_limit,
            record_progress=record_progress,
            verbose=verbose,
        )
        self.warm_start_value = warm_start_value
        self.input_file_name = input_file_name
        self.s = s
    
    def _fit_solver_specific(self, X, y):
        """
        Fit the solver specific parameters and prepare the input file for the solver.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values.
        """
        if self.time_limit is None:
            # Set a default time limit if not provided. Practically infinity.
            self.time_limit = 1<<30
        if self.warm_start_value is None:
            # Set a default warm start value if not provided. Practically infinity.
            self.warm_start_value = 1<<30
        n = len(X)
        m= len(X[0])
        deltal, deltar = self.__parse_attack_model(self.attack_model)
        deltal1 = [str(x) for x in deltal]
        deltar2 = [str(x) for x in deltar]

        args = []
        for x in X:
            args.extend(x)
        args.extend(y)
        args =  list(map(str, args))  # Convert to strings
        
        preamble = [n,m,*deltal1,*deltar2,self.time_limit,self.max_depth,self.s,1e-7,self.warm_start_value,int(self.record_progress)]
        preambleStr = list(map(str,preamble))
        # write the input file
        with open("exhaustivesearch/"+self.input_file_name, "w") as f:
            f.write("\n".join(preambleStr+args))

        self._construct_tree()
    def _construct_tree(self):
        """
        Construct the tree from the solver output.
        This method runs the solver and constructs the tree based on the output.
        """
        binary_path = os.path.abspath("solver.exe")
        input_path = os.path.abspath("exhaustivesearch/"+self.input_file_name)
        # Run the solver with the input file
        result = subprocess.run([binary_path, input_path], capture_output=True, text=True)
        resultstr = result.stdout.splitlines()
        selected_depth = int(resultstr[0])
        resultstr = resultstr[1:]
        self.optimal_=bool(resultstr[-1])
        nodes = [None for _ in range(2**(selected_depth+1))]
        number_of_nodes = 2**(selected_depth+1)-1
        for i in range(number_of_nodes)[::-1]:
            if i >= 2**(selected_depth) - 1:
                classification = int(resultstr[i][1])
                nodes[i] =  Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, np.array([1-classification, classification]))
            else:
                dim, thresh=(resultstr[i].split())
                dim = int(dim)
                thresh = float(thresh)
                nodes[i] = NumericalNode(
                    dim, thresh+1e-8, nodes[2*i+1], nodes[2*i+2], _TREE_UNDEFINED
                )
        self.root_ = nodes[0]
        os.remove("exhaustivesearch/"+self.input_file_name)
        
        if self.record_progress:
            resultstr = resultstr[number_of_nodes:]
            self.runtimes_ = []
            self.upper_bounds_ = []
            for elem in resultstr:
                if elem == "***":
                    break
                # timestamp returned as millisecond
                # error at time stamp is equal to number of misclassified samples
                timestamp, error_at_timestamp = list(map(int,elem.split()))
                self.runtimes_.append(timestamp/1000)
                self.upper_bounds_.append(error_at_timestamp)



    

    def __parse_attack_model(self, attack_model):
        """
        Parse the attack model into two lists Delta_l and Delta_r.
        Parameters
        ----------
        attack_model : list
            The attack model to parse.

        Returns
        -------
        tuple
            A tuple containing the two lists Delta_l and Delta_r.
        """
        Delta_l = []
        Delta_r = []
        for attack in attack_model:
            if isinstance(attack, Number):
                Delta_l.append(attack)
                Delta_r.append(attack)
            elif isinstance(attack, tuple):
                Delta_l.append(attack[0])
                Delta_r.append(attack[1])
            elif attack == "":
                Delta_l.append(0)
                Delta_r.append(0)
            elif attack == ">":
                Delta_l.append(0)
                Delta_r.append(1)
            elif attack == "<":
                Delta_l.append(1)
                Delta_r.append(0)
            elif attack == "<>":
                Delta_l.append(1)
                Delta_r.append(1)
        return Delta_l, Delta_r



        


