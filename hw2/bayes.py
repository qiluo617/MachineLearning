import numpy as np
z = np.genfromtxt('spambase.data', dtype=float, delimiter=',')
np.random.seed(0)
rp = np.random.permutation(z.shape[0])
z = z[rp, :]
x = z[:, :-1]
y = z[:, -1]

train_size = 2000

# print(x.shape)

for i in range(x.shape[1]):
    median = np.median(x[:,i])
    # print(median)
    for j in range(x.shape[0]):
        if x[j, i] > median:
            x[j, i] = 1
        else:
            x[j, i] = 0

train_data = x[:train_size, :]
test_data = x[train_size:, :]

train_y = y[:train_size]
test_y = y[train_size:]

prob_p0 = np.ones(len(train_data[0]))
sum_p0 = len(train_data[0])
prob_p1 = np.ones(len(train_data[0]))
sum_p1 = len(train_data[0])

for i in range(len(train_y)):
    if train_y[i] == 0:
        prob_p0 += train_data[i]
        sum_p0 += sum(train_data[i])
    else:
        prob_p1 += train_data[i]
        sum_p1 += sum(train_data[i])

p1 = np.dot(test_data, np.log(prob_p1 / sum_p1)) + np.log(sum(train_y) / float(len(train_y)))
p0 = np.dot(test_data, np.log(prob_p0 / sum_p0)) + np.log(1 - sum(train_y) / float(len(train_y)))

result = np.zeros(len(p0))

for i in range(len(p0)):
    if p1[i] < p0[i]:
        result[i] = 0
    else:
        result[i] = 1

correct = 0
correct_one = 0
for i in range(len(test_y)):
    if result[i] == test_y[i]:
        correct += 1
    if np.ones(len(test_y))[i] == test_y[i]:
        correct_one += 1

acc = correct / len(test_y)
acc_one = correct_one / len(test_y)

print(1-acc)
print(1-acc_one)
