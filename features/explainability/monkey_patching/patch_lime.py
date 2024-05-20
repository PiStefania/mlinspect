from typing import Any, List

import gorilla
import numpy as np
from lime import lime_tabular

from features.explainability.backends._explainability_backend import (
    ExplainabilityBackend,
)

from mlinspect import (
    BasicCodeLocation,
    CodeReference,
    DagNode,
    DagNodeDetails,
    FunctionInfo,
    OperatorContext,
    OperatorType,
)
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._monkey_patching_utils import (
    add_dag_node,
    add_test_data_dag_node,
    execute_patched_func_indirect_allowed,
    execute_patched_func_no_op_id,
    get_dag_node_for_id,
    get_optional_code_info_or_none,
)


class LimeCallInfo:
    param_search_active: bool = False
    mlinspect_explainer_node_id: int = -1
    parent_nodes: List[DagNode] | None = None
    actual_explainer_input: Any | None = None


call_info_singleton_lime = LimeCallInfo()


class LimePatching:
    """Patches for pandas"""

    # pylint: disable=too-few-public-methods

    @gorilla.patch(
        lime_tabular.LimeTabularExplainer,
        name="__init__",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched__init__(
        self,
        training_data: np.ndarray,
        mode: str = "classification",
        training_labels: Any | None = None,
        feature_names: List[str] | None = None,
        categorical_features: List[int] | None = None,
        categorical_names: Any | None = None,
        kernel_width: float | None = None,
        kernel: Any | None = None,
        verbose: bool = False,
        class_names: List[str] | None = None,
        feature_selection: str = "auto",
        discretize_continuous: bool = True,
        discretizer: str = "quartile",
        sample_around_instance: bool = False,
        random_state: int | Any | None = None,
        training_data_stats: dict | None = None,
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str | None = None,
        mlinspect_fit_transform_active: bool = False,
    ) -> Any:
        """Patch for ('lime.lime_tabular', 'LimeTabularExplainer')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(
            lime_tabular.LimeTabularExplainer, "__init__"
        )

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = (
            mlinspect_optional_code_reference
        )
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_non_data_func_args = {
            "training_data": training_data,
            "mode": mode,
            "training_labels": training_labels,
            "feature_names": feature_names,
            "categorical_features": categorical_features,
            "categorical_names": categorical_names,
            "kernel_width": kernel_width,
            "kernel": kernel,
            "verbose": verbose,
            "class_names": class_names,
            "feature_selection": feature_selection,
            "discretize_continuous": discretize_continuous,
            "discretizer": discretizer,
            "sample_around_instance": sample_around_instance,
            "random_state": random_state,
            "training_data_stats": training_data_stats,
        }

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> None:
            """Execute inspections, add DAG node"""
            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code
            call_info_singleton_lime.mlinspect_explainer_node_id = (
                singleton.get_next_op_id()
            )

            function_info = FunctionInfo(
                "lime.lime_tabular.LimeTabularExplainer", "__init__"
            )
            data_backend_result, test_data_node, _ = add_test_data_dag_node(
                training_data,
                function_info,
                lineno,
                optional_code_reference,
                optional_source_code,
                caller_filename,
            )

            operator_context = OperatorContext(
                OperatorType.CREATE_EXPLAINER, function_info
            )

            input_dfs = [data_backend_result.annotated_dfobject]
            input_infos = ExplainabilityBackend().before_call(
                operator_context, input_dfs
            )
            result = original(self, **self.mlinspect_non_data_func_args)
            backend_result = ExplainabilityBackend().after_call(
                operator_context,
                input_infos,
                result,
                self.mlinspect_non_data_func_args,
            )

            dag_node = DagNode(
                call_info_singleton_lime.mlinspect_explainer_node_id,
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails("LIME Explainer", []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            add_dag_node(dag_node, [test_data_node], backend_result)

        return execute_patched_func_no_op_id(
            original,
            execute_inspections,
            self,
            **self.mlinspect_non_data_func_args
        )

    @gorilla.patch(
        lime_tabular.LimeTabularExplainer,
        name="explain_instance",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched_explain_instance(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('lime_tabular.LimeTabularExplainer', 'explain_instance')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            lime_tabular.LimeTabularExplainer, "explain_instance"
        )

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> Any:
            """Execute inspections, add DAG node"""
            # pylint: disable=too-many-locals
            function_info = FunctionInfo(
                "lime.lime_tabular.LimeTabularExplainer", "explain_instance"
            )
            # Test data
            data_backend_result, test_data_node, test_data_result = (
                add_test_data_dag_node(
                    args[0],
                    function_info,
                    lineno,
                    optional_code_reference,
                    optional_source_code,
                    caller_filename,
                )
            )

            operator_context = OperatorContext(
                OperatorType.EXPLAINABILITY, function_info
            )
            input_dfs = [data_backend_result.annotated_dfobject]
            input_infos = ExplainabilityBackend().before_call(
                operator_context, input_dfs
            )
            call_info_singleton_lime.actual_explainer_input = test_data_result
            result = original(self, *args, **kwargs)
            backend_result = ExplainabilityBackend().after_call(
                operator_context,
                input_infos,
                result,
                self.mlinspect_non_data_func_args,
            )
            dag_node = DagNode(
                singleton.get_next_op_id(),
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails("LIME", []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )

            explainer_dag_node = get_dag_node_for_id(
                call_info_singleton_lime.mlinspect_explainer_node_id
            )
            parent_nodes = [explainer_dag_node, test_data_node]
            if call_info_singleton_lime.parent_nodes:
                parent_nodes.extend(call_info_singleton_lime.parent_nodes)
            add_dag_node(
                dag_node,
                parent_nodes,
                backend_result,
            )
            return result

        if not call_info_singleton_lime.param_search_active:
            new_result = execute_patched_func_indirect_allowed(
                execute_inspections
            )
        else:
            original = gorilla.get_original_attribute(
                lime_tabular.LimeTabularExplainer, "explain_instance"
            )
            new_result = original(self, *args, **kwargs)
        return new_result
