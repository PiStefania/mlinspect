"""
The Interface for the different instrumentation backends
"""

import abc
import dataclasses
from types import MappingProxyType
from typing import Any, Dict, List

from mlinspect import OperatorContext
from mlinspect.inspections import Inspection


@dataclasses.dataclass(frozen=True)
class AnnotatedDfObject:
    """A dataframe-like object and its annotations"""

    result_data: Any
    result_annotation: Any


@dataclasses.dataclass(frozen=True)
class BackendResult:
    """The annotated dataframe and the annotations for the current DAG node"""

    annotated_dfobject: AnnotatedDfObject
    dag_node_annotation: Dict[Inspection, Any]
    optional_second_annotated_dfobject: AnnotatedDfObject | None = None
    optional_second_dag_node_annotation: Dict[Inspection, Any] | None = None


class Backend(metaclass=abc.ABCMeta):
    """
    The Interface for the different instrumentation backends
    """

    @abc.abstractmethod
    def before_call(
        self,
        operator_context: OperatorContext,
        input_infos: List[AnnotatedDfObject],
    ) -> List[AnnotatedDfObject]:
        """The value or module a function may be called on"""
        # pylint: disable=too-many-arguments, unused-argument
        raise NotImplementedError

    @abc.abstractmethod
    def after_call(
        self,
        operator_context: OperatorContext,
        input_infos: List[AnnotatedDfObject],
        return_value: Any,
        non_data_function_args: MappingProxyType = MappingProxyType({}),
    ) -> BackendResult:
        """The return value of some function"""
        # pylint: disable=too-many-arguments, unused-argument
        raise NotImplementedError
