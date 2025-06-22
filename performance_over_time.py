import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from exhaustivesearch.RobTree import OptimalRobustTreeSearch

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.2)
from groot.toolbox import Model

from roct.milp import OptimalRobustTree, BinaryOptimalRobustTree
from roct.maxsat import SATOptimalRobustTree

from groot.model import GrootTreeClassifier

# Avoid type 3 fonts
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

def compute_average_runtime_cost(many_runtimes, many_costs):
    all_runtimes = [item for sublist in many_runtimes for item in sublist]
    all_runtimes = np.sort(np.unique(all_runtimes))

    costs_resampled = []
    for runtimes, costs in zip(many_runtimes, many_costs):
        indices = np.searchsorted(runtimes, all_runtimes, side='right') - 1
        costs = np.array(costs)
        costs_resampled.append(costs[indices])

    mean_costs = np.sum(costs_resampled, axis=0) / len(many_runtimes)
    sem_costs = np.std(costs_resampled, axis=0, ddof=1) / np.sqrt(len(many_runtimes))
    return all_runtimes, mean_costs, sem_costs

def plot_runtimes_cost(many_runtimes, many_costs, color_index, label, only_avg=False):
    mean_runtimes, mean_costs, sem_costs = compute_average_runtime_cost(
        many_runtimes, many_costs
    )
    if only_avg:
        plt.fill_between(mean_runtimes, mean_costs, mean_costs + sem_costs, color=sns.color_palette()[color_index], alpha=0.05)
        plt.fill_between(mean_runtimes, mean_costs, mean_costs - sem_costs, color=sns.color_palette()[color_index], alpha=0.05)
    else:
        for runtimes, costs in zip(many_runtimes, many_costs):
            plt.plot(
                runtimes, costs, drawstyle="steps-post", c=sns.color_palette()[color_index], alpha=0.2
            )
    plt.plot(mean_runtimes, mean_costs, c=sns.color_palette()[color_index], drawstyle="steps-post", label=label)
    

depth = 2
time_limit = 60*10
use_cached = True

data_dir = "data/"
figure_dir = "out/figures/"
output_dir = "out/"

algorithms = [
    "lsu-maxsat",
    "milp-warm",
    "RobTree",
    "RobTree-warm",
    "Pure Search",
    "Pure Search-warm"
]
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
    "banknote-authentication": [0.07, 0.09, 0.11],
    "blood-transfusion": [0.01, 0.02, 0.03],
    "breast-cancer": [0.28, 0.39, 0.45],
    "cylinder-bands": [0.23, 0.28, 0.45],
    "diabetes": [0.05, 0.07, 0.09],
    "haberman": [0.02, 0.03, 0.05],
    "ionosphere": [0.2, 0.28, 0.36],
    "wine": [0.02, 0.03, 0.04],
}

print("running algorithms...")
if use_cached:
    with open(output_dir + "progress.txt") as file:


        milp_warm_runtimes = eval(file.readline())
        milp_warm_costs = eval(file.readline())



        lsu_sat_runtimes = eval(file.readline())
        lsu_sat_costs = eval(file.readline())

        quantrnb_runtimes = eval(file.readline())
        quantrnb_costs = eval(file.readline())

        quantrnb_warm_runtimes = eval(file.readline())
        quantrnb_warm_costs = eval(file.readline())

        robtree_runtimes = eval(file.readline())
        robtree_costs = eval(file.readline())

        robtree_warm_runtimes = eval(file.readline())
        robtree_warm_costs = eval(file.readline())

else:


    milp_warm_runtimes = []
    milp_warm_costs = []



    lsu_sat_runtimes = []
    lsu_sat_costs = []

    quantrnb_runtimes = []
    quantrnb_costs = []

    quantrnb_warm_runtimes = []
    quantrnb_warm_costs = []

    
    robtree_runtimes= []
    robtree_costs = []

    robtree_warm_runtimes= []
    robtree_warm_costs = []

    for dataset in datasets:
        # Load dataset samples
        X_train = np.load(data_dir + f"X_train_{dataset}.npy")
        X_test = np.load(data_dir + f"X_test_{dataset}.npy")

        # Load dataset labels
        y_train = np.load(data_dir + f"y_train_{dataset}.npy")
        y_test = np.load(data_dir + f"y_test_{dataset}.npy")

        epsilon = epsilons[dataset][1]
        attack_model = [epsilon] * X_train.shape[1]
        


        print("Running MILP-warm...")
        # MILP-warm
        groot_tree = GrootTreeClassifier(
             max_depth=depth, attack_model=attack_model, min_samples_split=2, random_state=1
         )
        groot_tree.fit(X_train, y_train)
        tree = OptimalRobustTree(
        attack_model=attack_model,
        max_depth=depth,
        time_limit=time_limit,
        record_progress=True,
        warm_start_tree=groot_tree,
        )
        tree.fit(X_train, y_train)
        milp_warm_runtimes.append([0.0] + tree.runtimes_)
        milp_warm_costs.append([1.0] + [cost / len(X_train) for cost in tree.upper_bounds_])




        print("Running LSU-SAT...")
        # LSU-SAT
        tree = SATOptimalRobustTree(
            attack_model=attack_model,
            max_depth=depth,
            record_progress=True,
            lsu=True,
            time_limit=time_limit,
        )
        tree.fit(X_train, y_train)
        lsu_sat_runtimes.append([0.0] + tree.runtimes_ + [max(time_limit, tree.runtimes_[-1])])
        lsu_sat_costs.append([1.0] + [cost / len(X_train) for cost in tree.upper_bounds_] + [tree.upper_bounds_[-1] / len(X_train)])


        print("Running RobTree...")
        # # QuantRNB
        tree = OptimalRobustTreeSearch(
        max_depth=depth,
        attack_model=attack_model,
        time_limit=time_limit,
        cpus=1,
        record_progress=True,
        verbose=False,
        construct_complete_tree=False,
        s = 3,
        input_file_name= "input.txt",
        warm_start_value=1<<30,
        )
        tree.fit(X_train, y_train)
        quantrnb_runtimes.append([0.0]+tree.runtimes_[1:] + [max(time_limit, tree.runtimes_[-1])])
        quantrnb_costs.append([1.0]+[cost / len(X_train) for cost in tree.upper_bounds_[1:]] + [tree.upper_bounds_[-1] / len(X_train)])

        print("Running RobTree-warm")
        # # QuantRNB-warm
        groot_tree = GrootTreeClassifier(
        max_depth=2, attack_model=attack_model, min_samples_split=2, random_state=1
        )
        groot_tree.fit(X_train, y_train)
        groot_tree = Model.from_groot(groot_tree)
        groot_adv = groot_tree.adversarial_accuracy(X_train, y_train, attack = "tree", epsilon=epsilon)
        warm_value = len(y_train) - np.ceil(len(y_train)*groot_adv)+1
    
        tree = OptimalRobustTreeSearch(
             max_depth=depth,
             attack_model=attack_model,
             time_limit=time_limit,
             record_progress=True,
             s = 3,
             input_file_name= "input.txt",
             warm_start_value=warm_value
        )
        tree.fit(X_train, y_train)
        quantrnb_warm_runtimes.append(tree.runtimes_ + [max(time_limit, tree.runtimes_[-1])])
        quantrnb_warm_costs.append([cost / len(X_train) for cost in tree.upper_bounds_] + [tree.upper_bounds_[-1] / len(X_train)])



        print("Running Pure Search")
        # QuantRNB
        tree = OptimalRobustTreeSearch(
        max_depth=depth,
        attack_model=attack_model,
        time_limit=time_limit,
        cpus=1,
        record_progress=True,
        verbose=False,
        construct_complete_tree=False,
        s = 1<<30,
        input_file_name= "input.txt",
        warm_start_value=1<<30,
        )
        tree.fit(X_train, y_train)
        robtree_runtimes.append([0.0]+tree.runtimes_[1:] + [max(time_limit, tree.runtimes_[-1])])
        robtree_costs.append([1.0]+[cost / len(X_train) for cost in tree.upper_bounds_[1:]] + [tree.upper_bounds_[-1] / len(X_train)])

        print("Running Pure Search-warm")
        # QuantRNB-warm
        groot_tree = GrootTreeClassifier(
        max_depth=2, attack_model=attack_model, min_samples_split=2, random_state=1
        )
        groot_tree.fit(X_train, y_train)
        groot_tree = Model.from_groot(groot_tree)
        groot_adv = groot_tree.adversarial_accuracy(X_train, y_train, attack = "tree", epsilon=epsilon)
        warm_value = len(y_train) - np.ceil(len(y_train)*groot_adv)+1
    
        tree = OptimalRobustTreeSearch(
            max_depth=depth,
            attack_model=attack_model,
            time_limit=time_limit,
            record_progress=True,
            s = 1<<30,
            input_file_name= "input.txt",
            warm_start_value=warm_value
        )
        tree.fit(X_train, y_train)
        robtree_warm_runtimes.append(tree.runtimes_ + [max(time_limit, tree.runtimes_[-1])])
        robtree_warm_costs.append([cost / len(X_train) for cost in tree.upper_bounds_] + [tree.upper_bounds_[-1] / len(X_train)])

    print("Writing to file...")
    with open(output_dir + "progress.txt", "w") as file:
        



        file.write(str(milp_warm_runtimes) + '\n')
        file.write(str(milp_warm_costs) + '\n')

        file.write(str(lsu_sat_runtimes) + '\n')
        file.write(str(lsu_sat_costs) + '\n')

        file.write(str(quantrnb_runtimes) + '\n')
        file.write(str(quantrnb_costs) + '\n')

        file.write(str(quantrnb_warm_runtimes) + '\n')
        file.write(str(quantrnb_warm_costs) + '\n')

        file.write(str(robtree_runtimes) + '\n')
        file.write(str(robtree_costs) + '\n')

        file.write(str(robtree_warm_runtimes) + '\n')
        file.write(str(robtree_warm_costs) + '\n')

print("Preparing Results...")
plot_runtimes_cost(milp_warm_runtimes, milp_warm_costs, 1, "MILP-warm")
plot_runtimes_cost(lsu_sat_runtimes, lsu_sat_costs, 2, "LSU-MaxSAT")
plot_runtimes_cost(quantrnb_runtimes, quantrnb_costs, 3, "RobTree")
plot_runtimes_cost(quantrnb_warm_runtimes, quantrnb_warm_costs, 4, "RobTree-warm")
plot_runtimes_cost(robtree_runtimes, robtree_costs, 5, "Pure Search")
plot_runtimes_cost(robtree_warm_runtimes, robtree_warm_costs, 6, "Pure Search-warm")




plt.xlim(0.1, time_limit)
plt.xlabel("Time (s)")
plt.ylabel("% training error")
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(figure_dir + "cost_over_time.png")
plt.savefig(figure_dir + "cost_over_time.pdf")
plt.close()

plot_runtimes_cost(milp_warm_runtimes, milp_warm_costs, 1, "MILP-warm", only_avg=True)
plot_runtimes_cost(lsu_sat_runtimes, lsu_sat_costs, 2, "LSU-MaxSAT", only_avg=True)
plot_runtimes_cost(quantrnb_runtimes, quantrnb_costs, 3, "RobTree", only_avg=True)
plot_runtimes_cost(quantrnb_warm_runtimes, quantrnb_warm_costs, 4, "RobTree-warm", only_avg=True)
plot_runtimes_cost(robtree_runtimes, robtree_costs, 5, "Pure Search", only_avg=True)
plot_runtimes_cost(robtree_warm_runtimes, robtree_warm_costs, 6, "Pure Search-warm", only_avg=True)



plt.xlim(0.1, time_limit)
plt.xlabel("Time (s)")
plt.ylabel("Mean % training error")
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(figure_dir + "mean_cost_over_time.png")
plt.savefig(figure_dir + "mean_cost_over_time.pdf")
plt.close()
