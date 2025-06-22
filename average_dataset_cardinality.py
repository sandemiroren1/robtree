
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


import numpy as np

dataset_to_unique = {}
for datasetname in datasets:
    X = np.load("data/X_train_"+datasetname+".npy")

    res = 0
    print(X[0].shape)
    for feat in range(X[0].shape[0]):
        feature_sets = set()
        for x in X:
            feature_sets.add(x[feat])
        res+=len(feature_sets)
    dataset_to_unique[datasetname]=res/X[0].shape[0]
print(dataset_to_unique)
    




