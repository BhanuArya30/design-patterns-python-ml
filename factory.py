from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class ModelFactory:
    @staticmethod
    def get_model(model_type):
        if model_type == "logistic":
            return LogisticRegression()
        elif model_type == "svm":
            return SVC()
        elif model_type == "random_forest":
            return RandomForestClassifier()
        else:
            raise ValueError(f"Model type {model_type} not recognized.")


# Data preparation
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

# Factory usage
model = ModelFactory.get_model("random_forest")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Accuracy of Random Forest: {accuracy_score(y_test, predictions)}")
