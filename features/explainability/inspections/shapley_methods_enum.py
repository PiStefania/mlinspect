from enum import Enum


class ShapleyMethodsEnum(Enum):
    SHAP = "SHAP"

    @classmethod
    def _missing_(cls, value):
        raise Exception("Shapley calculation method is not supported.")
