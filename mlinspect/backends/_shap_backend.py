from types import MappingProxyType
from typing import List, Dict

from mlinspect import OperatorType
from mlinspect.backends._backend import Backend, AnnotatedDfObject, BackendResult
from mlinspect.backends._pandas_backend import execute_inspection_visits_data_source
from mlinspect.backends._sklearn_backend import execute_inspection_visits_nary_op


class ShapBackend(Backend):
    @staticmethod
    def before_call(operator_context, input_infos) -> List[AnnotatedDfObject]:
        return input_infos

    @staticmethod
    def after_call(operator_context, input_infos: List[AnnotatedDfObject], return_value,
                   non_data_function_args: Dict[str, any] = MappingProxyType({})) -> BackendResult:
        if operator_context.operator == OperatorType.EXPLAINABILITY:
            return_value = execute_inspection_visits_nary_op(operator_context,
                                                             input_infos,
                                                             return_value,
                                                             non_data_function_args)
        if operator_context.operator == OperatorType.CREATE_EXPLAINER:
            return_value = execute_inspection_visits_data_source(operator_context,
                                                             return_value,
                                                             non_data_function_args)
        return return_value
