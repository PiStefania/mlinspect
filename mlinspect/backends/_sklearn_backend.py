"""
The scikit-learn backend
"""

from types import MappingProxyType
from typing import Any, List

import pandas

from .. import OperatorContext, OperatorType
from ..instrumentation._pipeline_executor import singleton
from ._backend import AnnotatedDfObject, Backend, BackendResult
from ._iter_creation import (
    iter_input_annotation_output_nary_op,
    iter_input_annotation_output_sink_op,
)
from ._pandas_backend import (
    execute_inspection_visits_data_source,
    execute_inspection_visits_unary_operator,
    store_inspection_outputs,
)


class SklearnBackend(Backend):
    """
    The scikit-learn backend
    """

    def before_call(
        self,
        operator_context: OperatorContext,
        input_infos: List[AnnotatedDfObject],
    ) -> List[AnnotatedDfObject]:
        """The value or module a function may be called on"""
        # pylint: disable=too-many-arguments
        if operator_context.operator == OperatorType.TRAIN_TEST_SPLIT:
            pandas_df = input_infos[0].result_data
            assert isinstance(pandas_df, pandas.DataFrame)
            pandas_df["mlinspect_index"] = range(0, len(pandas_df))
        return input_infos

    def after_call(
        self,
        operator_context: OperatorContext,
        input_infos: List[AnnotatedDfObject],
        return_value: Any | None,
        non_data_function_args: MappingProxyType | Any = MappingProxyType({}),
    ) -> BackendResult:
        """The return value of some function"""
        # pylint: disable=too-many-arguments
        if operator_context.operator == OperatorType.DATA_SOURCE:
            return_value_be = execute_inspection_visits_data_source(
                operator_context, return_value, non_data_function_args
            )
        elif operator_context.operator == OperatorType.TRAIN_TEST_SPLIT:
            train_data, test_data = return_value
            train_return_value = execute_inspection_visits_unary_operator(
                operator_context,
                input_infos[0].result_data,
                input_infos[0].result_annotation,
                train_data,
                True,
                non_data_function_args,
            )
            test_return_value = execute_inspection_visits_unary_operator(
                operator_context,
                input_infos[0].result_data,
                input_infos[0].result_annotation,
                test_data,
                True,
                non_data_function_args,
            )
            input_infos[0].result_data.drop(
                "mlinspect_index", axis=1, inplace=True
            )
            train_data.drop("mlinspect_index", axis=1, inplace=True)
            test_data.drop("mlinspect_index", axis=1, inplace=True)
            return_value_be = BackendResult(
                train_return_value.annotated_dfobject,
                train_return_value.dag_node_annotation,
                test_return_value.annotated_dfobject,
                test_return_value.dag_node_annotation,
            )
        elif operator_context.operator in {
            OperatorType.PROJECTION,
            OperatorType.PROJECTION_MODIFY,
            OperatorType.TRANSFORMER,
            OperatorType.TRAIN_DATA,
            OperatorType.TRAIN_LABELS,
            OperatorType.TEST_DATA,
            OperatorType.TEST_LABELS,
        }:
            return_value_be = execute_inspection_visits_unary_operator(
                operator_context,
                input_infos[0].result_data,
                input_infos[0].result_annotation,
                return_value,
                False,
                non_data_function_args,
            )
        elif operator_context.operator == OperatorType.ESTIMATOR:
            return_value_be = execute_inspection_visits_sink_op(
                operator_context,
                input_infos[0].result_data,
                input_infos[0].result_annotation,
                input_infos[1].result_data,
                input_infos[1].result_annotation,
                non_data_function_args,
            )
        elif operator_context.operator == OperatorType.SCORE:
            return_value_be = execute_inspection_visits_nary_op(
                operator_context,
                input_infos,
                return_value,
                non_data_function_args,
            )
        elif operator_context.operator == OperatorType.PREDICT:
            return_value_be = execute_inspection_visits_nary_op(
                operator_context,
                input_infos,
                return_value,
                non_data_function_args,
            )
        elif operator_context.operator == OperatorType.CONCATENATION:
            return_value_be = execute_inspection_visits_nary_op(
                operator_context,
                input_infos,
                return_value,
                non_data_function_args,
            )
        else:
            raise NotImplementedError(
                "SklearnBackend doesn't know any operations of type '{}' yet!".format(
                    operator_context.operator
                )
            )
        return return_value_be


# -------------------------------------------------------
# Execute inspections functions
# -------------------------------------------------------


def execute_inspection_visits_sink_op(
    operator_context: OperatorContext,
    data: Any,
    data_annotation: Any,
    target: Any,
    target_annotation: Any,
    non_data_function_args: Any,
) -> BackendResult:
    """Execute inspections"""
    # pylint: disable=too-many-arguments
    inspection_count = len(singleton.inspections)
    iterators_for_inspections = iter_input_annotation_output_sink_op(
        inspection_count,
        data,
        data_annotation,
        target,
        target_annotation,
        operator_context,
        non_data_function_args,
    )
    annotation_iterators = execute_visits(iterators_for_inspections)
    return_value = store_inspection_outputs(annotation_iterators, None)
    return return_value


def execute_inspection_visits_nary_op(
    operator_context: OperatorContext,
    annotated_dfs: List[AnnotatedDfObject],
    return_value_df: Any,
    non_data_function_args: Any,
) -> BackendResult:
    """Execute inspections"""
    # pylint: disable=too-many-arguments
    inspection_count = len(singleton.inspections)
    iterators_for_inspections = iter_input_annotation_output_nary_op(
        inspection_count,
        annotated_dfs,
        return_value_df,
        operator_context,
        non_data_function_args,
    )
    annotation_iterators = execute_visits(iterators_for_inspections)
    return_value = store_inspection_outputs(
        annotation_iterators, return_value_df
    )
    return return_value


def execute_visits(iterators_for_inspections: Any) -> list:
    """
    After creating the iterators we need depending on the operator type, we need to execute the
    generic inspection visits
    """
    annotation_iterators = []
    for inspection_index, inspection in enumerate(singleton.inspections):
        iterator_for_inspection = iterators_for_inspections[inspection_index]
        annotations_iterator = inspection.visit_operator(
            iterator_for_inspection
        )
        annotation_iterators.append(annotations_iterator)
    return annotation_iterators
