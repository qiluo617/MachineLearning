import phate
import pandas as pd

train_data = pd.read_csv("train_data.csv", header=None)
train_label = pd.read_csv("train_label.csv", header=None)

test_data = pd.read_csv("test_data.csv", header=None)
test_label = pd.read_csv("test_label.csv", header=None)


def mnist(data, label, t):
    if t == 'auto':
        phate_operator = phate.PHATE()
    else:
        phate_operator = phate.PHATE(t=t)

    phate_operator.fit(data)
    mnist_phate = phate_operator.transform(plot_optimal_t=True)

    phate.plot.scatter2d(mnist_phate, c=label)


# train
#find optimal t
mnist(train_data, train_label, t='auto')

# t = 5 less than optimal t=25
mnist(train_data, train_label, t=5)

# t = 35 larger than optimal t=25
mnist(train_data, train_label, t=35)

# test
#find optimal t
mnist(test_data, test_label, t='auto')

# t = 5 less than optimal t=25
mnist(test_data, test_label, t=5)

# t = 35 larger than optimal t=25
mnist(test_data, test_label, t=35)
