import numpy as np

def take_from(block: np.ndarray, p: float) -> np.ndarray:
    """
    Take first p% of the block.
    Args:
        block (np.ndarray): The block to take from.
        p (float): The percentage to take.
    Returns:
        np.ndarray: The block with the first p% taken.
    """
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    n = int(len(block) * p)
    return block[:n]

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def loss(y, y_hat):
    m = y.shape[0]
    return -np.sum(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15)) / m

def dw(y, y_hat, X):
    m = y.shape[0]
    return X.T @ (y_hat - y) / m

class Z:
    def __init__(self, features, labels):
        self.features = np.hstack((np.ones((features.shape[0], 1)), features))
        self.labels = labels.reshape(-1, 1)
        self.weights = np.zeros((self.features.shape[1], 1))
        # self.__log_weights = []      # use if need to save log
        # self.__log_loss = []

    def __repr__(self):
        return f"Z(features size={self.features.shape}, labels size={self.labels.shape}, weights shape={self.weights.shape})"

    def train(self, learning_rate=2, epochs=10000, decrease_lr=True):
        s = epochs // 6
        previous_loss = float('inf')
        for epoch in range(epochs + 1):
            if decrease_lr and epoch % s == 0 and epoch != 0:
                learning_rate *= 0.5
            z = self.features @ self.weights
            y_hat = sigmoid(z)
            loss_value = loss(self.labels, y_hat)
            if abs(loss_value - previous_loss) < 1e-8:
                print(f"Loss converged at epoch {epoch}.")
                break
            previous_loss = loss_value
            self.weights -= learning_rate * dw(self.labels, y_hat, self.features)
            if (epoch/epochs)*100 % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss_value:.4f}")
                # self.__log_loss.append([epoch, loss_value])
                # self.__log_weights.append([epoch, self.weights.copy()])
        print(f"Training completed after {epoch} epochs.")

class MultinomialLogistic:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.learning_rate = 0.1
        self.epochs = 10000
        self.categories = np.unique(labels)
        self.models = self.__train_all_models()

    def __train_all_models(self):
        models = {}
        for category in self.categories:
            binary_labels = (self.labels == category).astype(float)
            model = Z(self.features, binary_labels)
            models[category] = model
        return models

    def train(self, learning_rate=0.1, epochs=10000, decrease_lr=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        for category, model in self.models.items():
            print(f"Training for category {category}...")
            model.train(learning_rate=self.learning_rate, epochs=self.epochs, decrease_lr=decrease_lr)
        print("Training completed for all categories.")

    def softmax(self, x):
        x = np.hstack((np.ones((1, 1)), x))  # add bias term
        scores = {c: sigmoid(x @ self.models[c].weights)[0, 0] for c in self.categories}
        total = sum(np.exp(v) for v in scores.values())
        softmax_scores = {c: np.exp(v) / total for c, v in scores.items()}
        return softmax_scores

    def predict(self, x):
        probs = self.softmax(x.reshape(1, -1))
        return max(probs, key=probs.get), round(probs[max(probs, key=probs.get)] * 100, 2)

    def accuracy(self, test_features, test_labels):
        correct = 0
        predict =[]
        for x, y in zip(test_features, test_labels):
            result, prob = self.predict(x)
            predict.append([result, prob])
            if result == y:
                correct += 1
        acc = correct / len(test_labels)
        print(f"Accuracy: {acc * 100:.2f}%")
        return acc, np.array(predict)

    def to_file(self, filename):
        data = {category: model.weights for category, model in self.models.items()}
        data_str_keys = {str(key): value for key, value in data.items()}
        np.savez(filename, **data_str_keys)