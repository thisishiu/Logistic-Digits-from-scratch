import argparse
from tqdm import tqdm
from scripts import *
from  process_image import *

parser = argparse.ArgumentParser(description="Train logistic regression on GPU with CuPy")
parser.add_argument('--gpu', action='store_true', help="Use GPU for training")
parser.add_argument('--p', type=float, default=0.8, help="Percentage of data to use for training")
parser.add_argument('--lr', type=float, default=8, help="Learning rate for training")
parser.add_argument('--epochs', type=int, default=10000, help="Number of epochs for training")
args = parser.parse_args()

# Check if GPU is available
if args.gpu:
    import cupy as cp
    xp = cp
    print("Using GPU (CuPy)")
else:
    import numpy as np
    xp = np
    print("Using CPU (NumPy)")

def read_idx(filename)-> xp.ndarray:
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
        data = xp.frombuffer(buffer=f.read(), dtype=data_type).reshape(size)
        return data

# Read MNIST data
images_1 = read_idx('data/t10k-images.idx3-ubyte')
labels_1 = read_idx('data/t10k-labels.idx1-ubyte')
images_2 = read_idx('data/train-images.idx3-ubyte')
labels_2 = read_idx('data/train-labels.idx1-ubyte')
images = xp.concatenate((images_1, images_2), axis=0)
labels = xp.concatenate((labels_1, labels_2), axis=0)
print(f"Length of images: {len(images)}")
print(f"Size of a single image: {images[0].shape}")
print(f"Length of labels: {len(labels)}")
print(f"Size of a single label: {labels[0].shape}")

# Re-center images
print("Re-centering images...")
for i in tqdm(range(len(images)), desc="Re-centering images"):
    images[i] = re_center_image(images[i], size=28, xp=xp)
    
print("Re-centering completed.")

# Take p% of the data
p = args.p
X = take_from(images, p)
X = X.reshape(X.shape[0], -1)
X = X / 255.0       # <---  Important step (normalize pixel values)
Y = take_from(labels, p)
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

# Instantiate and train the model
A = MultinomialLogistic(X, Y, xp)
print(f"Training with {p*100}% of the data")
A.train(learning_rate=args.lr, 
        epochs=args.epochs)
A.to_file('model')


# Test the model
print(f"Testing with {100-p*100}% of the data")
X_test = take_from(images, 1-p, reverse=True)
Y_test = take_from(labels, 1-p, reverse=True)
X_test = X_test.reshape(X_test.shape[0], -1)
X_test = X_test / 255.0
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")
count_true = 0
for i in tqdm(range(len(Y_test)), desc="Testing model"):
    y_hat, prob = A.predict(X_test[i])
    if y_hat == Y_test[i]:
        count_true += 1
print(f"Accuracy: {count_true/len(Y_test)*100:.2f}%")
