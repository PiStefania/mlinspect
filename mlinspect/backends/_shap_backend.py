from types import MappingProxyType
from typing import List, Dict

from mlinspect.backends._backend import Backend, AnnotatedDfObject, BackendResult
from mlinspect.backends._iter_creation import iter_input_annotation_output_nary_op
from mlinspect.backends._pandas_backend import store_inspection_outputs
from mlinspect.instrumentation._pipeline_executor import singleton


class ShapBackend(Backend):
    def before_call(self, operator_context, input_infos: List[AnnotatedDfObject]) -> List[AnnotatedDfObject]:
        print("before_call")
        return input_infos

    def after_call(self, operator_context, input_infos: List[AnnotatedDfObject], return_value,
                   non_data_function_args: Dict[str, any] = MappingProxyType({})) -> BackendResult:
        return_value = execute_inspection_visits_nary_op(operator_context,
                                                         input_infos,
                                                         return_value,
                                                         non_data_function_args)
        print("after_call")
        return return_value
def execute_inspection_visits_nary_op(operator_context, annotated_dfs: List[AnnotatedDfObject],
                                      return_value_df, non_data_function_args) -> BackendResult:
    """Execute inspections"""
    # pylint: disable=too-many-arguments
    inspection_count = len(singleton.inspections)
    iterators_for_inspections = iter_input_annotation_output_nary_op(inspection_count,
                                                                     annotated_dfs,
                                                                     return_value_df,
                                                                     operator_context,
                                                                     non_data_function_args)
    annotation_iterators = execute_visits(iterators_for_inspections)
    return_value = store_inspection_outputs(annotation_iterators, return_value_df)
    return return_value

def execute_visits(iterators_for_inspections):
    """
    After creating the iterators we need depending on the operator type, we need to execute the
    generic inspection visits
    """
    annotation_iterators = []
    for inspection_index, inspection in enumerate(singleton.inspections):
        iterator_for_inspection = iterators_for_inspections[inspection_index]
        annotations_iterator = inspection.visit_operator(iterator_for_inspection)
        annotation_iterators.append(annotations_iterator)
    return annotation_iterators