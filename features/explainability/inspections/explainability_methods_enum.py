from enum import Enum


class ExplainabilityMethodsEnum(Enum):
    SHAP = "Shapley Values"
    LIME = "Lime"
    PDP = "Partial Dependence Plots"
    ICE = "Individual Conditional Expectation"
    INTEGRATED_GRADIENTS = "Integrated Gradients"
    ALE = "Accumulated Local Effects"
    DALE = "Differential Accumulated Local Effects"
    DALEX = "Descriptive Machine Learning Explanations"

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise Exception("Explainability method is not supported.")
