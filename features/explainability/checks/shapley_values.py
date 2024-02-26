"""
The NoBiasIntroducedFor check
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, List

import numpy as np

from features.explainability.inspections.explainability_methods_enum import (
    ExplainabilityMethodsEnum,
)
from features.explainability.inspections.explainer import Explainer

from mlinspect import OperatorType
from mlinspect.checks._check import Check, CheckResult, CheckStatus
from mlinspect.inspections._inspection import Inspection
from mlinspect.inspections._inspection_result import InspectionResult


@dataclasses.dataclass
class ShapleyValues(CheckResult):
    values: np.ndarray | list | dict


class ShapleyValuesCheck(Check):
    # pylint: disable=unnecessary-pass, too-few-public-methods

    def __init__(
        self,
        methods: List[ExplainabilityMethodsEnum],
        explainer_input: np.ndarray,
        test_input: np.ndarray,
        test_labels: list,
        train_labels: list,
        features: list,
        nsamples: int = 100,
    ) -> None:
        self.explainer_input = explainer_input
        self.test_input = test_input
        self.nsamples = nsamples
        self.methods = methods
        self.features = features
        self.test_labels = test_labels
        self.train_labels = train_labels

    @property
    def check_id(self) -> Any | None:
        """The id of the Check"""
        return tuple([obj.name for obj in self.methods])

    @property
    def required_inspections(self) -> Iterable[Inspection]:
        """The inspections required for the check"""
        return [
            Explainer(
                self.methods,
                self.explainer_input,
                self.test_input,
                self.features,
                self.test_labels,
                self.train_labels,
                self.nsamples,
            )
        ]

    def evaluate(self, inspection_result: InspectionResult) -> CheckResult:
        """Evaluate the check"""
        ...
        dag = inspection_result.dag
        check_status = CheckStatus.SUCCESS
        description = None
        results = {}
        for (
            dag_node,
            inspection_results,
        ) in inspection_result.dag_node_to_inspection_results.items():
            results[dag_node] = inspection_results[
                Explainer(
                    self.methods,
                    self.explainer_input,
                    self.test_input,
                    self.features,
                    self.test_labels,
                    self.train_labels,
                    self.nsamples,
                )
            ]
        relevant_node = [
            node
            for node in dag.nodes
            if node.operator_info.operator
            in {
                OperatorType.ESTIMATOR,
            }
        ][0]
        results = results[relevant_node]
        return ShapleyValues(self, check_status, description, results)
