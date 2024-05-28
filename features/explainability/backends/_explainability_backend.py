from types import MappingProxyType
from typing import Any, List

from mlinspect import OperatorContext, OperatorType
from mlinspect.backends._backend import (
    AnnotatedDfObject,
    Backend,
    BackendResult,
)
from mlinspect.backends._sklearn_backend import (
    execute_inspection_visits_nary_op,
)


class ExplainabilityBackend(Backend):
    def before_call(
        self,
        operator_context: OperatorContext,
        input_infos: List[AnnotatedDfObject],
    ) -> List[AnnotatedDfObject]:
        return input_infos

    def after_call(
        self,
        operator_context: OperatorContext,
        input_infos: List[AnnotatedDfObject],
        return_value: Any,
        non_data_function_args: MappingProxyType | Any = MappingProxyType({}),
    ) -> BackendResult:
        if operator_context.operator == OperatorType.EXPLAINABILITY:
            return_value_be = execute_inspection_visits_nary_op(
                operator_context,
                input_infos,
                return_value,
                non_data_function_args,
            )
        elif operator_context.operator == OperatorType.CREATE_EXPLAINER:
            return_value_be = execute_inspection_visits_nary_op(
                operator_context,
                input_infos,
                return_value,
                non_data_function_args,
            )
        else:
            raise NotImplementedError(
                "ExplainabilityBackend doesn't know any operations of type '{}' yet!".format(
                    operator_context.operator
                )
            )
        return return_value_be
