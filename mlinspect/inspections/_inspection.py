"""
The Interface for the Inspection
"""

import abc
from typing import Any, Iterable, Union

from ._inspection_input import (
    InspectionInputDataSource,
    InspectionInputNAryOperator,
    InspectionInputSinkOperator,
    InspectionInputUnaryOperator,
)


class Inspection(metaclass=abc.ABCMeta):
    """
    The Interface for the Inspections
    """

    @property
    def inspection_id(self) -> Any | None:
        """The id of the inspection"""
        return None

    @abc.abstractmethod
    def visit_operator(
        self,
        inspection_input: Union[
            InspectionInputDataSource,
            InspectionInputUnaryOperator,
            InspectionInputNAryOperator,
            InspectionInputSinkOperator,
        ],
    ) -> Iterable[Any]:
        """Visit an operator in the DAG"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_operator_annotation_after_visit(self) -> Any:
        """Get the output to be included in the DAG"""
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        """Inspections must implement equals"""
        return (
            isinstance(other, self.__class__)
            and self.inspection_id == other.inspection_id
        )

    def __hash__(self) -> int:
        """Inspections must be hashable"""
        return hash((self.__class__.__name__, self.inspection_id))

    def __repr__(self) -> str:
        """Inspections must have a str representation"""
        return "{}({})".format(self.__class__.__name__, self.inspection_id)
