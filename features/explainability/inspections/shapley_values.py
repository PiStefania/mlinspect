"""
The ShapleyValues Inspection
"""
import dataclasses
from _pydecimal import Decimal
from typing import Iterable, Dict

from mlinspect import FunctionInfo, OperatorType
from mlinspect.inspections import Inspection, InspectionInputUnaryOperator


@dataclasses.dataclass(frozen=True, eq=True)
class ShapleyValuesInfo:
    values: Dict[str, Decimal]


class ShapleyValues(Inspection):

    def __init__(self):
        self._operator_type = None
        self._output_values = []

    def visit_operator(self, inspection_input) -> Iterable[any]:
        """
        Visit an operator
        """
        self._output_values = []
        self._operator_type = inspection_input.operator_context.operator

        if self._operator_type == OperatorType.EXPLAINABILITY:
            for row in inspection_input.row_iterator:
                self._output_values.append(row.output)
        else:
            for _ in inspection_input.row_iterator:
                yield None


    def get_operator_annotation_after_visit(self) -> any:
        assert self._operator_type
        if self._operator_type is not OperatorType.ESTIMATOR:
            result = self._output_values
            self._output_values = []
            return result
        self._operator_type = None
        self._output_values = []
        return None

    @property
    def inspection_id(self):
        return None
