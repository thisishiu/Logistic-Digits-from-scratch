def take_from(block, p, reverse=False):
    """
    Take first p% of the block.
    Args:
        block (np.ndarray): The block to take from.
        p (float): The percentage to take.
    Returns:
        np.ndarray: The block with be p% of data.
    """
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    n = int(len(block) * p)
    return block[:n] if not reverse else block[-n:]

def sigmoid(z, xp):
    z = xp.clip(z, -500, 500)
    return 1 / (1 + xp.exp(-z))

def loss(y, y_hat, xp):
    m = y.shape[0]
    return -xp.sum(y * xp.log(y_hat + 1e-15) + (1 - y) * xp.log(1 - y_hat + 1e-15)) / m

def dw(y, y_hat, X):
    m = y.shape[0]
    return X.T @ (y_hat - y) / m

class Z:
    def __init__(self, features, labels, xp):
        self.xp = xp
        self.features = xp.hstack((xp.ones((features.shape[0], 1)), features))
        self.labels = labels.reshape(-1, 1)
        self.weights = xp.zeros((self.features.shape[1], 1))
        # self.__log_weights = []      # use if need to save log
        # self.__log_loss = []

    def __repr__(self):
        return f"Z(features size={self.features.shape}, labels size={self.labels.shape}, weights shape={self.weights.shape})"

    def train(self, learning_rate=2, epochs=10000):
        print(f"Training with learning rate: {learning_rate}, epochs: {epochs} ({'GPU' if self.xp.__name__ == 'cupy' else 'CPU'})")
        # s = epochs // 2
        previous_loss = float('inf')
        for epoch in range(epochs + 1):
            # if epoch % s == 0 and epoch != 0:
            #     learning_rate *= 0.5
            z = self.features @ self.weights
            y_hat = sigmoid(z, self.xp)
            loss_value = loss(self.labels, y_hat, self.xp)
            if loss_value > previous_loss:
                learning_rate *= 0.5
            previous_loss = loss_value
            self.weights -= learning_rate * dw(self.labels, y_hat, self.features)
            if (epoch/epochs)*100 % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss_value:.4f}")
                # self.__log_loss.append([epoch, loss_value])
                # self.__log_weights.append([epoch, self.weights.copy()])
        print(f"Training completed after {epoch} epochs.")

class MultinomialLogistic:
    def __init__(self, features, labels, xp):
        self.xp = xp
        self.features = features
        self.labels = labels
        self.categories = xp.unique(labels)
        self.models = self.__train_all_models()

    def __train_all_models(self):
        models = {}
        for category in self.categories:
            binary_labels = (self.labels == category).astype(float)
            model = Z(self.features, binary_labels, self.xp)
            models[int(category)] = model
        return models

    def train(self, learning_rate=0.1, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        for category, model in self.models.items():
            print(f"Training for category {category}...")
            model.train(learning_rate=self.learning_rate, epochs=self.epochs)
        print("Training completed for all categories.")

    def softmax(self, x):
        x = self.xp.hstack((self.xp.ones((1, 1)), x))  # add bias term
        scores = {c: sigmoid(x @ self.models[c].weights, self.xp)[0, 0] for c in self.categories}
        total = sum(self.xp.exp(v) for v in scores.values())
        softmax_scores = {c: self.xp.exp(v) / total for c, v in scores.items()}
        return softmax_scores

    def predict(self, x):
        probs = self.softmax(x.reshape(1, -1))
        return max(probs, key=probs.get), round(probs[max(probs, key=probs.get)] * 100, 2)

    def to_file(self, filename):
        data = {category: model.weights for category, model in self.models.items()}
        if self.xp.__name__ == "cupy":
            data_str_keys = {str(key): value.get() for key, value in data.items()}
        else:
            data_str_keys = {str(key): value for key, value in data.items()}
        self.xp.savez_compressed(filename, **data_str_keys)