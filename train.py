import numpy as np
from scripts import *

def read_idx(filename)-> np.ndarray:
    dataType = {
            8	:'uint8',
            9	:'int8',
            11	:'int16',
            12	:'int32',
            13	:'float32',
            14	:'float64'
            }
    with open(filename, 'rb') as f:
        magic = f.read(4)
        data_type = dataType[magic[2]]
        dim = magic[3]
        size = []
        for d in range(dim):
            size.append(int.from_bytes(f.read(4), 'big', signed=False))
        data = np.frombuffer(buffer=f.read(), dtype=data_type).reshape(size)
        return data

images_1 = read_idx('data/t10k-images.idx3-ubyte')
labels_1 = read_idx('data/t10k-labels.idx1-ubyte')
images_2 = read_idx('data/train-images.idx3-ubyte')
labels_2 = read_idx('data/train-labels.idx1-ubyte')
images = np.concatenate((images_1, images_2), axis=0)
labels = np.concatenate((labels_1, labels_2), axis=0)
print(f"Length of images: {len(images)}")
print(f"Size of a single image: {images[0].shape}")
print(f"Length of labels: {len(labels)}")
print(f"Size of a single label: {labels[0].shape}")

p = 1
X = take_from(images, p)
X = X.reshape(X.shape[0], -1)
X = X / 255.0       # <---  Important step (normalize pixel values)
Y = take_from(labels, p)
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

A = MultinomialLogistic(X, Y)
A.train(0.05, 10000)  #<---- Can modify
A.to_file('model')


