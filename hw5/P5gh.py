import phate
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

train_data = pd.read_csv("train_data.csv", header=None)
train_label = pd.read_csv("train_label.csv", header=None)

test_data = pd.read_csv("test_data.csv", header=None)
test_label = pd.read_csv("test_label.csv", header=None)


def phate_mean(data, label, shape):

    phate_operator = phate.PHATE(n_components=10)

    phate_operator.fit(data)
    mnist_phate = phate_operator.transform(plot_optimal_t=True)

    print(mnist_phate.shape)

    diff_potential = phate_operator.diff_potential
    y_pred = KMeans(n_clusters=10).fit_predict(diff_potential)

    true = label.to_numpy()

    #ARI
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

    print(np.average(ARI))

    phate_operator_plot = phate.PHATE(t=25)
    phate_operator_plot.fit(data)
    mnist_phate_plot = phate_operator_plot.transform()
    phate.plot.scatter2d(mnist_phate_plot, c=y_pred)


# train
phate_mean(train_data, train_label, 60000)

# test
phate_mean(test_data, test_label, 10000)
