"""
The ShapleyValues Inspection
"""
from typing import Iterable

import numpy as np
from scikeras.wrappers import KerasClassifier

from features.explainability.inspections.shapley_methods_enum import ShapleyMethodsEnum
from mlinspect import OperatorType
from mlinspect.inspections import Inspection


class Explainer(Inspection):

    def __init__(self, method: ShapleyMethodsEnum, explainer_input: np.ndarray, test_input: np.ndarray, nsamples:int=100):
        self.method = method
        self._operator_type = None
        self._explainer = None
        self.shapley_values = None
        self.test_input = test_input
        self.nsamples = nsamples
        self.explainer_input = explainer_input

    def visit_operator(self, inspection_input) -> Iterable[any]:
        """
        Visit an operator
        """
        self._operator_type = inspection_input.operator_context.operator
        if self.method == ShapleyMethodsEnum.SHAP:
            import shap
            if self._operator_type == OperatorType.ESTIMATOR:
                for row in inspection_input.row_iterator:
                    model: KerasClassifier = row.output[0]
                    self._explainer = shap.KernelExplainer(model.predict, self.explainer_input)
                    self.shapley_values = self._explainer.shap_values(self.test_input, nsamples=self.nsamples)
        yield None


    def get_operator_annotation_after_visit(self) -> any:
        assert self._operator_type
        if self._operator_type is OperatorType.ESTIMATOR:
            result = [self._explainer, self.shapley_values]
            self._operator_type = None
            self._explainer = None
            self.shapley_values = None
            return result
        self._operator_type = None
        self._explainer = None
        self.shapley_values = None
        return None

    @property
    def inspection_id(self):
        return tuple([self.method])

