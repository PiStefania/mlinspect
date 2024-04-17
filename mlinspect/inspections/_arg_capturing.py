"""
A simple inspection to capture important function call arguments like estimator hyperparameters
"""

from typing import Any, Dict, Iterable, Union

from ._inspection import Inspection
from ._inspection_input import (
    InspectionInputDataSource,
    InspectionInputNAryOperator,
    InspectionInputSinkOperator,
    InspectionInputUnaryOperator,
)


class ArgumentCapturing(Inspection):
    """
    A simple inspection to capture important function call arguments like estimator hyperparameters
    """

    def __init__(self) -> None:
        self._captured_arguments: Dict[str, Any] | None = None

    @property
    def inspection_id(self) -> Any | None:
        return None

    def visit_operator(
        self,
        inspection_input: Union[
            InspectionInputDataSource,
            InspectionInputUnaryOperator,
            InspectionInputNAryOperator,
            InspectionInputSinkOperator,
        ],
    ) -> Iterable[Any]:
        """
        Visit an operator
        """
        self._captured_arguments = inspection_input.non_data_function_args

        for _ in inspection_input.row_iterator:
            yield None

    def get_operator_annotation_after_visit(self) -> Any:
        captured_args = self._captured_arguments
        self._captured_arguments = None
        return captured_args
