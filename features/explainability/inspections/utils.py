from scikeras.wrappers import KerasClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier


def is_supported_estimator(model):
    return isinstance(
        model,
        (
            KerasClassifier,
            DecisionTreeClassifier,
            LogisticRegression,
            SGDClassifier,
        ),
    )
