"""
The NoBiasIntroducedFor check
"""
from __future__ import annotations

import dataclasses
from typing import Iterable

import numpy as np

from features.explainability.inspections.explainer import Explainer
from features.explainability.inspections.shapley_methods_enum import ShapleyMethodsEnum
from mlinspect import OperatorType
from mlinspect.checks._check import Check, CheckStatus, CheckResult
from mlinspect.inspections._inspection import Inspection
from mlinspect.inspections._inspection_result import InspectionResult

@dataclasses.dataclass
class ShapleyValues(CheckResult):
   values: np.ndarray | list

class ShapleyValuesCheck(Check):
    # pylint: disable=unnecessary-pass, too-few-public-methods

    def __init__(self, method: ShapleyMethodsEnum, explainer_input: np.ndarray, test_input: np.ndarray, nsamples=100):
        self.explainer_input = explainer_input
        self.test_input = test_input
        self.nsamples = nsamples
        self.method = method

    @property
    def check_id(self):
        """The id of the Check"""
        return tuple([self.method, self.nsamples])

    @property
    def required_inspections(self) -> Iterable[Inspection]:
        """The inspections required for the check"""
        return [Explainer(self.method, self.explainer_input, self.test_input, self.nsamples)]

    def evaluate(self, inspection_result: InspectionResult) -> CheckResult:
        """Evaluate the check"""
        dag = inspection_result.dag
        check_status = CheckStatus.SUCCESS
        description = None
        results = {}
        for dag_node, inspection_results in inspection_result.dag_node_to_inspection_results.items():
            results[dag_node] = inspection_results[Explainer(self.method, self.explainer_input, self.test_input, self.nsamples)]
        relevant_node = [node for node in dag.nodes if node.operator_info.operator in {OperatorType.ESTIMATOR,}][0]
        results = results[relevant_node]
        values = results[1]
        return ShapleyValues(self, check_status, description, values)

