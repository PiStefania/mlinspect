"""
Monkey patching for numpy
"""

from typing import Any

import gorilla
from statsmodels import api
from statsmodels.api import datasets

from mlinspect import BasicCodeLocation, CodeReference, DagNode, DagNodeDetails
from mlinspect.backends._pandas_backend import PandasBackend
from mlinspect.backends._sklearn_backend import SklearnBackend
from mlinspect.inspections._inspection_input import (
    FunctionInfo,
    OperatorContext,
    OperatorType,
)
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._monkey_patching_utils import (
    add_dag_node,
    add_train_data_node,
    add_train_label_node,
    execute_patched_func,
    execute_patched_func_no_op_id,
    get_input_info,
    get_optional_code_info_or_none,
)


@gorilla.patches(api)
class StatsmodelApiPatching:
    """Patches for statsmodel"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("add_constant")
    @gorilla.settings(allow_hit=True)
    def patched_random(*args: Any, **kwargs: Any) -> Any:
        """Patch for ('statsmodel.api', 'add_constant')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(api, "add_constant")

        def execute_inspections(
            op_id: int,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> Any:
            """Execute inspections, add DAG node"""
            function_info = FunctionInfo("statsmodel.api", "add_constant")
            input_info = get_input_info(
                args[0],
                caller_filename,
                lineno,
                function_info,
                optional_code_reference,
                optional_source_code,
            )

            operator_context = OperatorContext(
                OperatorType.PROJECTION_MODIFY, function_info
            )
            input_infos = SklearnBackend().before_call(
                operator_context, [input_info.annotated_dfobject]
            )
            result = original(input_infos[0].result_data, *args[1:], **kwargs)
            backend_result = SklearnBackend().after_call(
                operator_context, input_infos, result
            )
            new_return_value = backend_result.annotated_dfobject.result_data

            dag_node = DagNode(
                op_id,
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails("Adds const column", ["array"]),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            add_dag_node(dag_node, [input_info.dag_node], backend_result)

            return new_return_value

        return execute_patched_func(
            original, execute_inspections, *args, **kwargs
        )


@gorilla.patches(datasets)
class StatsmodelsDatasetPatching:
    """Patches for pandas"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("get_rdataset")
    @gorilla.settings(allow_hit=True)
    def patched_read_csv(*args: Any, **kwargs: Any) -> Any:
        """Patch for ('statsmodels.datasets', 'get_rdataset')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(datasets, "get_rdataset")

        def execute_inspections(
            op_id: int,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> Any:
            """Execute inspections, add DAG node"""
            function_info = FunctionInfo(
                "statsmodels.datasets", "get_rdataset"
            )

            operator_context = OperatorContext(
                OperatorType.DATA_SOURCE, function_info
            )
            input_infos = PandasBackend().before_call(operator_context, [])
            result = original(*args, **kwargs)
            backend_result = PandasBackend().after_call(
                operator_context, input_infos, result.data
            )
            result.data = backend_result.annotated_dfobject.result_data
            dag_node = DagNode(
                op_id,
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails(result.title, list(result.data.columns)),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            add_dag_node(dag_node, [], backend_result)
            return result

        return execute_patched_func(
            original, execute_inspections, *args, **kwargs
        )


@gorilla.patches(api.OLS)
class StatsmodelsOlsPatching:
    """Patches for statsmodel OLS"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("__init__")
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('statsmodel.api', 'OLS')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals
        original = gorilla.get_original_attribute(api.OLS, "__init__")

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> None:
            """Execute inspections, add DAG node"""
            original(self, *args, **kwargs)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(
            original, execute_inspections, *args, **kwargs
        )

    @gorilla.name("fit")
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('statsmodel.api.OLS', 'fit')"""
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(api.OLS, "fit")
        function_info = FunctionInfo("statsmodel.api.OLS", "fit")

        # Train data
        # pylint: disable=no-member
        data_backend_result, train_data_node, train_data_result = (
            add_train_data_node(self, self.data.exog, function_info)  # type: ignore[attr-defined]
        )
        self.data.exog = train_data_result  # type: ignore[attr-defined]
        # pylint: disable=no-member
        label_backend_result, train_labels_node, train_labels_result = (
            add_train_label_node(self, self.data.endog, function_info)  # type: ignore[attr-defined]
        )
        self.data.endog = train_labels_result  # type: ignore[attr-defined]

        # Estimator
        operator_context = OperatorContext(
            OperatorType.ESTIMATOR, function_info
        )
        input_dfs = [
            data_backend_result.annotated_dfobject,
            label_backend_result.annotated_dfobject,
        ]
        input_infos = SklearnBackend().before_call(operator_context, input_dfs)
        result = original(self, *args, **kwargs)
        estimator_backend_result = SklearnBackend().after_call(
            operator_context, input_infos, None
        )

        dag_node = DagNode(
            singleton.get_next_op_id(),
            BasicCodeLocation(
                self.mlinspect_caller_filename, self.mlinspect_lineno
            ),
            operator_context,
            DagNodeDetails("Decision Tree", []),
            get_optional_code_info_or_none(
                self.mlinspect_optional_code_reference,
                self.mlinspect_optional_source_code,
            ),
        )
        add_dag_node(
            dag_node,
            [train_data_node, train_labels_node],
            estimator_backend_result,
        )
        return result
