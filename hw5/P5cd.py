import phate
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

train_data = pd.read_csv("train_data.csv", header=None)
train_label = pd.read_csv("train_label.csv", header=None)

test_data = pd.read_csv("test_data.csv", header=None)
test_label = pd.read_csv("test_label.csv", header=None)


def kmean(data, label, shape):
    true = label.to_numpy()

    k_digits = KMeans(n_clusters=10)
    y_pred = k_digits.fit_predict(data)

    true = true.reshape((shape, 1))
    y_pred = y_pred.reshape((shape, 1))

    combine = np.array((true, y_pred)).T
    combine = combine.reshape((shape, 2))

    np.random.shuffle(combine)

    subsampling = 20
    ARI = np.zeros(subsampling)
    gap = int(shape / subsampling)

    for it in range(subsampling):
        start = int(it * gap)
        end = int((it+1)*gap)
        x = combine[start:end, 0]
        y = combine[start:end, 1]
        ARI[it] = adjusted_rand_score(x, y)

    print(ARI)
    print(np.average(ARI))

    phate_operator = phate.PHATE(t=25)
    kmean_phate = phate_operator.fit_transform(data)

    phate.plot.scatter2d(kmean_phate, c=y_pred)


# train
kmean(train_data, train_label, 60000)

# test
kmean(test_data, test_label, 10000)
