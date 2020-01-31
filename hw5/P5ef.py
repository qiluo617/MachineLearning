import phate
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score

train_data = pd.read_csv("train_data.csv", header=None)
train_label = pd.read_csv("train_label.csv", header=None)

test_data = pd.read_csv("test_data.csv", header=None)
test_label = pd.read_csv("test_label.csv", header=None)


def spectral(data, label, gamma_):

    data_shape = data.shape
    label_shape = label.shape

    data = data.to_numpy()
    label = label.to_numpy()

    if gamma_ != 0:  # test
        spectral_cluster = SpectralClustering(n_clusters=10, gamma=gamma_)
        y_pred = spectral_cluster.fit_predict(data)

        true = np.squeeze(label)
        ARI = adjusted_rand_score(true, y_pred)

        print('Testing ARI: ', ARI)

        phate_operator = phate.PHATE(t=25)
        spectral_phate = phate_operator.fit_transform(data)
        phate.plot.scatter2d(spectral_phate, c=y_pred)

    else:  # train
        index = np.arange(data_shape[0])
        np.random.shuffle(index)

        sample = 10
        subsample_size = int(data_shape[0] / sample)

        ARI_subsample = np.zeros(sample)
        gamma_subsample = np.zeros(sample)

        for i in range(sample):
            print('current:', i)
            start = int(i*subsample_size)
            end = int((i+1)*subsample_size)

            # subsampleing
            selected_data = np.zeros((subsample_size, data_shape[1]))
            selected_label = np.zeros((subsample_size, label_shape[1]))

            location = 0
            for id in index[start:end]:
                selected_data[location,:] = data[id,:]
                selected_label[location] = label[id]
                location += 1

            best_ARI_i = 0
            best_gamma = 0
            best_y_pred = np.zeros((subsample_size, label_shape[1]))

            for gamma_value in np.arange(0, 2, 0.2):
                spectral_cluster = SpectralClustering(n_clusters=10, gamma=gamma_value)  # affinity:default ‘rbf’
                y_pred = spectral_cluster.fit_predict(selected_data)

                selected_label = np.squeeze(selected_label)

                current_ARI = adjusted_rand_score(selected_label, y_pred)

                if current_ARI > best_ARI_i:
                    best_ARI_i = current_ARI
                    best_gamma = gamma_value
                    best_y_pred = y_pred

            ARI_subsample[i] = best_ARI_i
            gamma_subsample[i] = best_gamma

            # plot phate for the last subsample
            if i == int(sample-1):
                phate_operator = phate.PHATE(t=25)
                spectral_phate = phate_operator.fit_transform(selected_data)
                phate.plot.scatter2d(spectral_phate, c=best_y_pred)

        print('ARI: ', ARI_subsample)
        print('The average ARI: ', np.average(ARI_subsample))
        print('Gamma: ', gamma_subsample)
        print('The average gamma: ', np.average(gamma_subsample))


# train
# spectral(train_data, train_label, 0)

# test
spectral(test_data, test_label, 1.6)

