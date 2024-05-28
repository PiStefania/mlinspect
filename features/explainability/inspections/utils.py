from scikeras.wrappers import KerasClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier


def is_supported_estimator(model) -> bool:
    """
    Check if the model is a supported estimator for explainability.
    Note: To add more supported estimators, add them to the supported_estimators tuple
    """
    supported_estimators = (
        KerasClassifier,
        DecisionTreeClassifier,
        LogisticRegression,
        SGDClassifier,
    )
    return isinstance(
        model,
        supported_estimators,
    )


def is_neural_network(model) -> bool:
    """
    Check if the model is a neural network.
    Note: To add more neural network models, add them to the supported_neural_networks tuple
    """
    supported_neural_networks = (KerasClassifier,)
    return isinstance(model, supported_neural_networks)


def is_regression(model) -> bool:
    """
    Check if the model is a regression model.
    Note: To add more regression models, add them to the supported_regression_models tuple
    """
    supported_regression_models = (LogisticRegression,)
    return isinstance(model, supported_regression_models)


def should_ignore_patch_predict(input_str: str) -> bool:
    """
    Ignore patching the predict method of the model and use the original predict method, according to the input string.
    Note: To add more conditions to ignore patching the predict method, add them to the if-elif-else block, as needed.
    """
    input_str = input_str.lower()
    if "score" in input_str:
        return True
    elif "shap_values" in input_str:
        return True
    elif "predict" in input_str and "Explainer".lower() not in input_str:
        return True
    elif "PartialDependenceDisplay".lower() in input_str:
        return True
    elif "dalex" in input_str:
        return True
    elif "fit" in input_str:
        return True
    return False


def should_ignore_patch_predict_proba(input_str: str) -> bool:
    """
    Ignore patching the predict_proba method of the model and use the original predict method, according to the input string.
    Note: To add more conditions to ignore patching the predict method, add them to the if-elif-else block, as needed.
    """
    input_str = input_str.lower()
    if "score" in input_str:
        return True
    elif "lime" in input_str:
        return True
    elif "ale" in input_str:
        return True
    elif "PartialDependenceDisplay".lower() in input_str:
        return True
    elif "dalex" in input_str:
        return True
    elif "fit" in input_str:
        return True
    return False


def is_shap_source_code(input_str: str) -> bool:
    """
    Check if the input string contains shap source code.
    """
    input_str = input_str.lower()
    return "shap" in input_str and "Explainer".lower() in input_str


def is_lime_source_code(input_str: str) -> bool:
    """
    Check if the input string contains lime source code.
    """
    input_str = input_str.lower()
    return "explain_instance" in input_str
