import numpy as np
import struct

images_path = '/Users/qiluo/Desktop/ML/homework5/train-images-idx3-ubyte'
labels_path = '/Users/qiluo/Desktop/ML/homework5/train-labels-idx1-ubyte'

# images_path = '/Users/qiluo/Desktop/ML/homework5/t10k-images-idx3-ubyte'
# labels_path = '/Users/qiluo/Desktop/ML/homework5/t10k-labels-idx1-ubyte'

with open(labels_path, 'rb') as lbpath:
    magic, n = struct.unpack('>II',
                             lbpath.read(8))
    labels = np.fromfile(lbpath,
                         dtype=np.uint8)
with open(images_path, 'rb') as imgpath:
    magic, num, rows, cols = struct.unpack(">IIII",
                                           imgpath.read(16))
    images = np.fromfile(imgpath,
                         dtype=np.uint8).reshape(len(labels), 784)


print('Dimentions: %s x %s' % (images.shape[0], images.shape[1]))

np.savetxt(fname='/Users/qiluo/Desktop/ML/homework5/train_data.csv', X=images, delimiter=',', fmt='%d')
np.savetxt(fname='/Users/qiluo/Desktop/ML/homework5/train_label.csv', X=labels, delimiter=',', fmt='%d')

# np.savetxt(fname='/Users/qiluo/Desktop/ML/homework5/test_data.csv', X=images, delimiter=',', fmt='%d')
# np.savetxt(fname='/Users/qiluo/Desktop/ML/homework5/test_label.csv', X=labels, delimiter=',', fmt='%d')