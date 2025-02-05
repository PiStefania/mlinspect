"""
Monkey patching for numpy
"""

from typing import Any

import gorilla
import numpy
from numpy import random

from mlinspect import BasicCodeLocation, CodeReference, DagNode, DagNodeDetails
from mlinspect.backends._sklearn_backend import SklearnBackend
from mlinspect.inspections._inspection_input import (
    FunctionInfo,
    OperatorContext,
    OperatorType,
)
from mlinspect.monkeypatching._monkey_patching_utils import (
    add_dag_node,
    execute_patched_func,
    get_optional_code_info_or_none,
)


@gorilla.patches(random)
class NumpyRandomPatching:
    """Patches for sklearn"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("random")
    @gorilla.settings(allow_hit=True)
    def patched_random(*args: Any, **kwargs: Any) -> Any:
        """Patch for ('numpy.random', 'random')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(random, "random")

        def execute_inspections(
            op_id: int,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> Any:
            """Execute inspections, add DAG node"""
            function_info = FunctionInfo("numpy.random", "random")
            operator_context = OperatorContext(
                OperatorType.DATA_SOURCE, function_info
            )
            input_infos = SklearnBackend().before_call(operator_context, [])
            result = original(*args, **kwargs)
            backend_result = SklearnBackend().after_call(
                operator_context, input_infos, result
            )

            dag_node = DagNode(
                op_id,
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails("random", ["array"]),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            add_dag_node(dag_node, [], backend_result)
            new_return_value = backend_result.annotated_dfobject.result_data
            return new_return_value

        return execute_patched_func(
            original, execute_inspections, *args, **kwargs
        )


@gorilla.patches(numpy)
class NumpyPatching:
    """Patches for sklearn"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("array")
    @gorilla.settings(allow_hit=True)
    def patched_array(*args: Any, **kwargs: Any) -> Any:
        """Patch for ('numpy', 'array')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(numpy, "array")

        def execute_inspections(
            op_id: int,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> Any:
            """Execute inspections, add DAG node"""
            function_info = FunctionInfo("numpy", "array")
            operator_context = OperatorContext(
                OperatorType.DATA_SOURCE, function_info
            )
            input_infos = SklearnBackend().before_call(operator_context, [])
            result = original(*args, **kwargs)
            backend_result = SklearnBackend().after_call(
                operator_context, input_infos, result
            )

            dag_node = DagNode(
                op_id,
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails("array", ["array"]),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            add_dag_node(dag_node, [], backend_result)
            new_return_value = backend_result.annotated_dfobject.result_data
            return new_return_value

        return execute_patched_func(
            original, execute_inspections, *args, **kwargs
        )
