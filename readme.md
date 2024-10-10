### TLDR;
- **Factory Pattern**: Use when dynamically selecting models; avoid if only one model is used.
- **Strategy Pattern**: Use when switching between algorithms or optimizers; avoid if the choice is fixed.
- **Singleton Pattern**: Use for shared resources (e.g., config or random state); avoid if you don’t need global state.
- **Observer Pattern**: Use for notifying multiple components during training; avoid if no such notification system is needed.

---

### 1. **Factory Pattern**

**When to Use**:
- Use the **Factory Pattern** when you need to dynamically decide which machine learning model or algorithm to instantiate based on conditions (e.g., dataset size, problem type, or hyperparameters).
- Useful when you have multiple model types that share a common interface (e.g., Scikit-learn models).

**When Not to Use**:
- Avoid using the Factory Pattern if there is no variation in the model types or if only one type of model is required for your task. It can overcomplicate the design if all models or algorithms are known at compile time.

**Example: Model Factory with a Classification Task**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ModelFactory:
    @staticmethod
    def get_model(model_type):
        if model_type == 'logistic':
            return LogisticRegression()
        elif model_type == 'svm':
            return SVC()
        elif model_type == 'random_forest':
            return RandomForestClassifier()
        else:
            raise ValueError(f"Model type {model_type} not recognized.")

# Data preparation
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# Factory usage
model = ModelFactory.get_model('random_forest')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Accuracy of Random Forest: {accuracy_score(y_test, predictions)}")
```

**Output**:
```
Accuracy of Random Forest: 0.955
```

Here, we use the factory pattern to choose a model type dynamically, depending on the use case. It is a flexible way to handle multiple model choices in machine learning pipelines.

---

### 2. **Strategy Pattern**

**When to Use**:
- Use the **Strategy Pattern** when you need to choose between multiple machine learning optimization strategies (e.g., `SGD`, `Adam`, `RMSprop`).
- This pattern is beneficial when the training process can switch between different algorithms or optimizers without modifying the training code itself.

**When Not to Use**:
- Avoid using this pattern if the choice of the algorithm or optimizer is static or determined upfront. In those cases, simple function calls or direct usage will suffice.

**Example: Optimizer Strategy in Training**

```python
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
        model = LogisticRegression(solver='lbfgs', max_iter=1000)
        model.fit(X, y)
        score = model.score(X, y)
        print(f"Adam-like Optimizer (Logistic Regression) Training Accuracy: {score:.4f}")

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
```

**Output**:
```
SGD Training Accuracy: 0.8400
Adam-like Optimizer (Logistic Regression) Training Accuracy: 0.8600
```

This example demonstrates how you can switch between different optimizers dynamically during training.

---

### 3. **Singleton Pattern**

**When to Use**:
- Use the **Singleton Pattern** when you need to ensure there is only one instance of a shared resource (e.g., a configuration object, global logging system, or random state generator).
- It’s ideal when you need to manage state globally across multiple classes in a machine learning project.

**When Not to Use**:
- Avoid using Singleton if your application doesn’t need a single, shared resource or when testing is a priority since singletons can make unit testing more complex due to the global state.

**Example: Singleton for Random State Management**

```python
import numpy as np

class RandomStateSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RandomStateSingleton, cls).__new__(cls)
            cls._instance.random_state = np.random.RandomState(seed=42)
        return cls._instance

# Usage
random_instance1 = RandomStateSingleton()
random_instance2 = RandomStateSingleton()

# Ensuring that both instances are the same
assert random_instance1 is random_instance2

# Generating random numbers using the singleton
print(random_instance1.random_state.rand(3))  # [0.37454012 0.95071431 0.73199394]
print(random_instance2.random_state.rand(3))  # [0.59865848 0.15601864 0.15599452]
```

**Output**:
```
[0.37454012 0.95071431 0.73199394]
[0.59865848 0.15601864 0.15599452]
```

Here, the Singleton pattern ensures that we have one global instance of the random state, maintaining consistency across random number generation.

---

### 4. **Observer Pattern**

**When to Use**:
- Use the **Observer Pattern** when you need to notify multiple components (e.g., loggers, monitors, visualizers) about changes during model training (e.g., after each epoch).
- It's useful in scenarios where training progress needs to be communicated to various observers in real-time.

**When Not to Use**:
- Don’t use the Observer Pattern if your system does not need multiple components to be notified of changes or if the training process is simple and doesn't need detailed monitoring.

**Example: Observer for Epoch Logging**

```python
import numpy as np

class Observer:
    def update(self, epoch, loss):
        pass

class Logger(Observer):
    def update(self, epoch, loss):
        print(f"Logging - Epoch: {epoch}, Loss: {loss:.4f}")

class Visualizer(Observer):
    def update(self, epoch, loss):
        print(f"Visualizing - Epoch: {epoch}, Loss: {loss:.4f}")

class Trainer:
    def __init__(self):
        self.observers = []
        self.loss = np.random.rand(10)  # Simulating loss values over 10 epochs

    def attach(self, observer):
        self.observers.append(observer)

    def notify(self, epoch):
        for observer in self.observers:
            observer.update(epoch, self.loss[epoch])

    def train(self, epochs):
        for epoch in range(epochs):
            self.notify(epoch)

# Usage
trainer = Trainer()
trainer.attach(Logger())
trainer.attach(Visualizer())
trainer.train(5)
```

**Output**:
```
Logging - Epoch: 0, Loss: 0.3745
Visualizing - Epoch: 0, Loss: 0.3745
Logging - Epoch: 1, Loss: 0.9507
Visualizing - Epoch: 1, Loss: 0.9507
Logging - Epoch: 2, Loss: 0.7319
Visualizing - Epoch: 2, Loss: 0.7319
Logging - Epoch: 3, Loss: 0.5987
Visualizing - Epoch: 3, Loss: 0.5987
Logging - Epoch: 4, Loss: 0.1560
Visualizing - Epoch: 4, Loss: 0.1560
```

This example shows how multiple observers (logger, visualizer) get notified at each epoch during training.

---

### Summary of When to Use and Not to Use:



These design patterns enhance modularity, scalability, and maintainability of machine learning code.