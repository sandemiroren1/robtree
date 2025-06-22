import matplotlib.pyplot as plt

from roct.upper_bound import maximum_adversarial_accuracy

import seaborn as sns

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=0.8)

import pandas as pd

import numpy as np

# Avoid type 3 fonts
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from tqdm import tqdm

import os

import json
result_tables_timeouts = []
result_tables_accuracies_test = [] 
result_tables_accuracies_train = [] 
def calculate_win(table_to_be_used,type_of_table):
			"""Calculate the number of wins for each algorithm in the table and save the aggregate scores."""
			rank_table_training = table_to_be_used.rank(axis=1, method="min", ascending=False)
			# print(rank_table)
			wins_df_training = (rank_table_training == 1).sum(axis=0)

			# Average rank
			mean_rank_df_training = rank_table_training.mean(axis=0)
			sem_rank_df_training = rank_table_training.sem(axis=0)

		

			# Summarize aggregate scores in a table
			mean_score_df_training = table_to_be_used.mean(axis=0)
			sem_score_df_training = table_to_be_used.sem(axis=0)
			agg_score_df_training = pd.concat((mean_score_df_training, sem_score_df_training, mean_rank_df_training, sem_rank_df_training, wins_df_training), axis=1)
			agg_score_df_training.columns = ["Mean adversarial accuracy", "Standard error adversarial accuracy", "Mean rank", "Standard error rank", "Number of wins"]
			agg_score_df_training.to_latex(figure_dir + f"aggregate_scores_{type_of_table}.tex", float_format="%.3f",escape=False)
def bold_min_values(row):
    """Apply bold formatting to minimum values in each row"""
    max_val = row.max()
    return row.apply(lambda x: f"\\textbf{{{x:.3f}}}" if x == max_val else f"{x:.3f}")

for i in range(1,6):
	result_dir = f"out/results/depth_2_trial_{i}/"
	figure_dir = "out/figures/"
	data_dir = "data/"

	results = []
	for result_name in tqdm(os.listdir(result_dir)):
					filename = result_dir + result_name

					with open(filename) as file:
									result = json.load(file)

					dataset, algorithm, epsilon = result_name[:-5].split("_")

					# Load datasets
					

					results.append(
									(
													dataset,
													epsilon,
													algorithm,
													result["train_accuracy"],
													result["train_adv_accuracy"],
													result["test_accuracy"],
													result["test_adv_accuracy"],
													result["time_taken"],
												
									)
					)

	columns = [
					"Dataset",
					"Epsilon",
					"Algorithm",
					"Train accuracy",
					"Train adversarial accuracy",
					"Test accuracy",
					"Test adversarial accuracy",
					"Time Taken",
	]
	result_df = pd.DataFrame(results, columns=columns)



	algorithm_names = {
					#"tree": "Decision Tree",
					#"treant": "TREANT",
					"groot": "GROOT",
					"lsu-maxsat": "LSU-MaxSAT",
					#"rc2-maxsat": "RC2-MaxSAT",
					#"bin-milp": "Binary-MILP",
					"milp-warm": "MILP-warm",
					#"bin-milp-warm": "Binary-MILP-warm",
					"Pure-Search": "\\bruteforce",
					"RobTree" : "\\myalg",
					"RobTree-warm" : "\\myalg-warm",
					"Pure-Search-warm": "\\bruteforce"
	}

	# source ~/venv/venv_with_python3.7/bin/activate
	result_df["Algorithm"] = result_df["Algorithm"].map(algorithm_names)

	

	df = result_df
	# Apply timeout logic
	df['Result'] = df['Time Taken'].apply(lambda t: t if t < 1800 else t)

	
	

# Apply the function to format the table
	if i==2:
		training_accuracy_table = result_df.copy()
		test_accuracy_table = result_df.copy()

		training_accuracy_table = training_accuracy_table.pivot_table(index=['Dataset', 'Epsilon'],
					columns='Algorithm',
					values='Train adversarial accuracy',
					aggfunc='first')
		formatted_table_training = training_accuracy_table.apply(bold_min_values, axis=1).drop(columns=["\myalg-warm,\bruteforce-warm"])
		formatted_table_training.to_latex("out/figures/training_accuracy.tex",escape=False)

		test_accuracy_table = test_accuracy_table.pivot_table(index=['Dataset', 'Epsilon'],
					columns='Algorithm',
					values="Test adversarial accuracy",
					aggfunc='first')
		formatted_table_test = test_accuracy_table.apply(bold_min_values, axis=1).drop(columns=["\myalg-warm,\bruteforce-warm"])
		formatted_table_test.to_latex("out/figures/test_accuracy.tex",escape=False)
		calculate_win(training_accuracy_table,"train")
		calculate_win(test_accuracy_table,"test")


		
	
	# Pivot the DataFrame: rows = Dataset + Epsilon, columns = Algorithm, values = Result
	summary_df = df.pivot_table(
					index=['Dataset', 'Epsilon'],
					columns='Algorithm',
					values='Result',
					aggfunc='first'
	).reset_index()

	# Optional: Fill missing algorithms with "timeout"
	summary_df = summary_df.fillna(999999)
	summary_df = summary_df.drop(columns=["GROOT"])

	result_tables_timeouts.append(summary_df)
	

	



grouped = pd.concat(result_tables_timeouts).groupby(level=0)

mean_timeouts =grouped.mean()
mean_timeouts_copy = mean_timeouts.copy()
mean_timeouts_copy=mean_timeouts_copy.applymap(lambda x : round(x) if x >=1 else 0)
rank_table = mean_timeouts_copy.rank(axis=1, method="min", ascending=True)
# print(rank_table)
wins_df = (rank_table == 1).sum(axis=0)

# Average rank
mean_rank_df = rank_table.mean(axis=0)
sem_rank_df = rank_table.sem(axis=0)

		

# Summarize aggregate scores in a table
mean_score_df = mean_timeouts_copy.mean(axis=0)
sem_score_df = mean_timeouts_copy.sem(axis=0)
agg_score_df = pd.concat((mean_score_df, sem_score_df, mean_rank_df, sem_rank_df, wins_df), axis=1)
agg_score_df.columns = ["Mean time taken", "Standard error time taken", "Mean rank", "Standard error rank", "Number of wins"]
agg_score_df.to_latex(figure_dir + f"aggregate_scores_timeout.tex", float_format="%.3f",escape=False)



print(mean_timeouts)

minima = (round(mean_timeouts.min(axis=1)))

# Restore non-numeric columns (e.g., 'Algorithm', 'Dataset', 'Epsilon')
metadata = grouped.first()[['Dataset', 'Epsilon']]
print(minima)
mean_timeouts = pd.concat([metadata, mean_timeouts], axis=1)
for i, row in mean_timeouts.iterrows():
					for col in mean_timeouts.columns:
									mean_timeouts[col] = mean_timeouts[col].astype('object')
									if col in ["Dataset", "Epsilon"]:
													continue  # Don't modify the index columns
									val = row[col]
									val_float = val
									if val_float > 1800 :
											mean_timeouts.at[i,col]= '-'
											continue
									if val_float <= 1:
											mean_timeouts.at[i,col]= "$\mathbf{<1}$"
											continue
									val_float = int(round(val_float))
									if abs(val_float-int(minima[i]))<0.0001:
													mean_timeouts.at[i, col] = f"$\\mathbf{{{(val_float)}}}$"
									else:
													mean_timeouts.at[i, col] = (val_float)



mean_timeouts.to_latex("out/figures/average_timeout.tex", escape=False)

print(mean_timeouts)


    


