import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

# Data preparation
np.random.seed(42)
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, size=100)


class Optimizer:
    def optimize(self, X, y):
        pass


class SGD(Optimizer):
    def optimize(self, X, y):
        model = SGDClassifier(max_iter=1000, tol=1e-3)
        model.fit(X, y)
        score = model.score(X, y)
        print(f"SGD Training Accuracy: {score:.4f}")


class Adam(Optimizer):
    def optimize(self, X, y):
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(X, y)
        score = model.score(X, y)
        print(
            f"Adam-like Optimizer (Logistic Regression) Training Accuracy: {score:.4f}"
        )


class TrainingPipeline:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def train(self, X, y):
        self.optimizer.optimize(X, y)


# Usage
optimizer = SGD()
pipeline = TrainingPipeline(optimizer)
pipeline.train(X_train, y_train)

# Switching to Adam-like Optimizer
optimizer = Adam()
pipeline = TrainingPipeline(optimizer)
pipeline.train(X_train, y_train)
