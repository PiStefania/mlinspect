from typing import Any, List

import dalex
import gorilla
import numpy as np

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
from mlinspect.backends._backend import AnnotatedDfObject
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._monkey_patching_utils import (
    add_dag_node,
    add_test_data_dag_node,
    add_test_label_node,
    execute_patched_func_indirect_allowed,
    execute_patched_func_no_op_id,
    get_dag_node_for_id,
    get_optional_code_info_or_none,
)


class DalexCallInfo:
    param_search_active: bool = False
    mlinspect_explainer_node_id: int = -1
    parent_nodes: List[DagNode] | None = None


call_info_singleton_dalex = DalexCallInfo()


class DalexPatching:
    """Patches for DALEX"""

    # pylint: disable=too-few-public-methods

    @gorilla.patch(
        dalex.Explainer,
        name="__init__",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched__init__(
        self,
        model: Any,
        data: Any,
        y: Any,
        predict_function: Any | None = None,
        residual_function: Any | None = None,
        weights: Any | None = None,
        label: str | None = None,
        model_class: str | None = None,
        verbose: bool = True,
        precalculate: bool = True,
        model_type: str | None = None,
        model_info: dict | None = None,
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str | None = None,
        mlinspect_fit_transform_active: bool = False,
    ) -> Any:
        """Patch for ('dalex', 'Explainer')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(dalex.Explainer, "__init__")

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = (
            mlinspect_optional_code_reference
        )
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_non_data_func_args = {
            "model": model,
            "data": data.view(np.ndarray),
            "y": y.view(np.ndarray),
            "predict_function": predict_function,
            "residual_function": residual_function,
            "weights": weights,
            "label": label,
            "model_class": model_class,
            "verbose": verbose,
            "precalculate": precalculate,
            "model_type": model_type,
            "model_info": model_info,
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
            call_info_singleton_dalex.mlinspect_explainer_node_id = (
                singleton.get_next_op_id()
            )

            function_info = FunctionInfo("dalex.Explainer", "__init__")

            operator_context = OperatorContext(
                OperatorType.CREATE_EXPLAINER, function_info
            )

            # Test data
            _, test_data_node, _ = add_test_data_dag_node(
                data,
                function_info,
                lineno,
                optional_code_reference,
                optional_source_code,
                caller_filename,
            )
            # Test labels
            _, test_labels_node, _ = add_test_label_node(
                y,
                caller_filename,
                function_info,
                lineno,
                optional_code_reference,
                optional_source_code,
            )

            input_infos = ExplainabilityBackend().before_call(
                operator_context, []
            )
            original(self, **self.mlinspect_non_data_func_args)
            result = self
            backend_result = ExplainabilityBackend().after_call(
                operator_context,
                input_infos,
                result,
                self.mlinspect_non_data_func_args,
            )

            dag_node = DagNode(
                call_info_singleton_dalex.mlinspect_explainer_node_id,
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails("DALEX Explainer", []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            parent_nodes = [test_data_node, test_labels_node]
            if call_info_singleton_dalex.parent_nodes:
                parent_nodes.extend(call_info_singleton_dalex.parent_nodes)
            add_dag_node(
                dag_node,
                parent_nodes,
                backend_result,
            )

        return execute_patched_func_no_op_id(
            original,
            execute_inspections,
            self,
            **self.mlinspect_non_data_func_args
        )

    @gorilla.patch(
        dalex.Explainer,
        name="model_parts",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched_model_parts(
        self,
        loss_function: str | None = None,
        type: str = "variable_importance",  # pylint: disable=redefined-builtin
        N: int = 1000,
        B: int = 10,
        variables: np.ndarray | str | None = None,
        variable_groups: List[dict] | None = None,
        keep_raw_permutations: bool = True,
        label: str | None = None,
        processes: int = 1,
        random_state: int | None = None,
        **kwargs: Any
    ) -> Any:
        """Patch for ('dalex.Explainer', 'model_parts')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            dalex.Explainer, "model_parts"
        )
        args = {
            "loss_function": loss_function,
            "type": type,
            "N": N,
            "B": B,
            "variables": variables,
            "variable_groups": variable_groups,
            "keep_raw_permutations": keep_raw_permutations,
            "label": label,
            "processes": processes,
            "random_state": random_state,
        }

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> Any:
            """Execute inspections, add DAG node"""
            # pylint: disable=too-many-locals
            function_info = FunctionInfo("dalex.Explainer", "model_parts")
            operator_context = OperatorContext(
                OperatorType.EXPLAINABILITY, function_info
            )
            input_dfs: List[AnnotatedDfObject] = []
            input_infos = ExplainabilityBackend().before_call(
                operator_context, input_dfs
            )
            result = original(self, **args, **kwargs)
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
                DagNodeDetails(args["type"], []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )

            explainer_dag_node = get_dag_node_for_id(
                call_info_singleton_dalex.mlinspect_explainer_node_id
            )
            add_dag_node(dag_node, [explainer_dag_node], backend_result)
            return result

        if not call_info_singleton_dalex.param_search_active:
            new_result = execute_patched_func_indirect_allowed(
                execute_inspections
            )
        else:
            original = gorilla.get_original_attribute(
                dalex.Explainer, "model_parts"
            )
            new_result = original(self, **args)
        return new_result

    @gorilla.patch(
        dalex.Explainer,
        name="predict_parts",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched_predict_parts(
        self,
        new_observation: Any,
        type: str = "break_down_interactions",  # pylint: disable=redefined-builtin
        order: List[int] | List[str] | None = None,
        interaction_preference: int = 1,
        path: str = "average",
        N: int | None = None,
        B: int = 25,
        keep_distributions: bool = False,
        label: str | None = None,
        processes: int = 1,
        random_state: int | None = None,
        **kwargs: Any
    ) -> Any:
        """Patch for ('dalex.Explainer', 'predict_parts')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            dalex.Explainer, "predict_parts"
        )
        args = {
            "new_observation": new_observation,
            "type": type,
            "N": N,
            "B": B,
            "order": order,
            "interaction_preference": interaction_preference,
            "path": path,
            "label": label,
            "processes": processes,
            "random_state": random_state,
            "keep_distributions": keep_distributions,
        }

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> Any:
            """Execute inspections, add DAG node"""
            # pylint: disable=too-many-locals
            function_info = FunctionInfo("dalex.Explainer", "predict_parts")
            operator_context = OperatorContext(
                OperatorType.EXPLAINABILITY, function_info
            )
            # Test data
            _, test_data_node, _ = add_test_data_dag_node(
                new_observation,
                function_info,
                lineno,
                optional_code_reference,
                optional_source_code,
                caller_filename,
            )
            input_dfs: List[AnnotatedDfObject] = []
            input_infos = ExplainabilityBackend().before_call(
                operator_context, input_dfs
            )
            result = original(self, **args, **kwargs)
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
                DagNodeDetails(args["type"], []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )

            explainer_dag_node = get_dag_node_for_id(
                call_info_singleton_dalex.mlinspect_explainer_node_id
            )
            add_dag_node(
                dag_node, [explainer_dag_node, test_data_node], backend_result
            )
            return result

        if not call_info_singleton_dalex.param_search_active:
            new_result = execute_patched_func_indirect_allowed(
                execute_inspections
            )
        else:
            original = gorilla.get_original_attribute(
                dalex.Explainer, "predict_parts"
            )
            new_result = original(self, **args)
        return new_result
