"""
A simple empty inspection
"""

from typing import Any, Iterable, Union

from mlinspect.inspections import (
    InspectionInputDataSource,
    InspectionInputNAryOperator,
    InspectionInputSinkOperator,
    InspectionInputUnaryOperator,
)
from mlinspect.inspections._inspection import Inspection


class EmptyInspection(Inspection):
    """
    An empty inspection for performance experiments
    """

    def __init__(self, inspection_id: Any) -> None:
        self._id = inspection_id

    @property
    def inspection_id(self) -> Any | None:
        return self._id

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
        for _ in inspection_input.row_iterator:
            yield None

    def get_operator_annotation_after_visit(self) -> Any:
        return None
