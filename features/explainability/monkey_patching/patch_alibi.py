from typing import Any, Callable, Dict, List, Union

import alibi
import gorilla
import numpy as np
import tensorflow as tf

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


class AlibiCallInfo:
    param_search_active: bool = False
    mlinspect_explainer_node_id: int = -1
    parent_nodes_ig: List[DagNode] | None = None
    parent_nodes_ale: List[DagNode] | None = None
    actual_explainer_input: Any | None = None


call_info_singleton_alibi = AlibiCallInfo()


class AlibiPatching:
    """Patches for alibi"""

    # pylint: disable=too-few-public-methods

    @gorilla.patch(
        alibi.explainers.IntegratedGradients,
        name="__init__",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched__init__(
        self,
        model: tf.keras.Model,
        layer: (
            Callable[[tf.keras.Model], tf.keras.layers.Layer]
            | tf.keras.layers.Layer
            | None
        ) = None,
        target_fn: Union[Callable, None] = None,
        method: str = "gausslegendre",
        n_steps: int = 50,
        internal_batch_size: int = 100,
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str | None = None,
        mlinspect_fit_transform_active: bool = False,
    ) -> Any:
        """Patch for ('alibi.explainers', 'IntegratedGradients')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(
            alibi.explainers.IntegratedGradients, "__init__"
        )

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = (
            mlinspect_optional_code_reference
        )
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_non_data_func_args = {
            "model": model,
            "layer": layer,
            "target_fn": target_fn,
            "method": method,
            "n_steps": n_steps,
            "internal_batch_size": internal_batch_size,
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
            call_info_singleton_alibi.mlinspect_explainer_node_id = (
                singleton.get_next_op_id()
            )

            function_info = FunctionInfo(
                "alibi.explainers.IntegratedGradients", "__init__"
            )

            operator_context = OperatorContext(
                OperatorType.CREATE_EXPLAINER, function_info
            )
            input_infos = ExplainabilityBackend().before_call(
                operator_context, []
            )
            result = original(self, **self.mlinspect_non_data_func_args)
            backend_result = ExplainabilityBackend().after_call(
                operator_context,
                input_infos,
                result,
                self.mlinspect_non_data_func_args,
            )

            dag_node = DagNode(
                call_info_singleton_alibi.mlinspect_explainer_node_id,
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails("Alibi Explainer", []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            parent_nodes = []
            if call_info_singleton_alibi.parent_nodes_ig:
                parent_nodes.extend(call_info_singleton_alibi.parent_nodes_ig)
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
        alibi.explainers.IntegratedGradients,
        name="explain",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched_explain(
        self,
        X: np.ndarray | List[np.ndarray],
        forward_kwargs: dict | None = None,
        baselines: (
            int
            | float
            | np.ndarray
            | List[int]
            | List[float]
            | List[np.ndarray]
            | None
        ) = None,
        target: int | list | np.ndarray | None = None,
        attribute_to_layer_inputs: bool = False,
    ) -> Any:
        """Patch for ('alibi.explainers.IntegratedGradients', 'explain')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            alibi.explainers.IntegratedGradients, "explain"
        )
        args = {
            "X": X,
            "forward_kwargs": forward_kwargs,
            "baselines": baselines,
            "target": target,
            "attribute_to_layer_inputs": attribute_to_layer_inputs,
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
            function_info = FunctionInfo(
                "alibi.explainers.IntegratedGradients", "explain"
            )
            # Test data
            data_backend_result, test_data_node, _ = add_test_data_dag_node(
                X,
                function_info,
                lineno,
                optional_code_reference,
                optional_source_code,
                caller_filename,
            )

            operator_context = OperatorContext(
                OperatorType.EXPLAINABILITY, function_info
            )
            input_dfs = [data_backend_result.annotated_dfobject]
            input_infos = ExplainabilityBackend().before_call(
                operator_context, input_dfs
            )

            result = original(self, **args)
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
                DagNodeDetails("Integrated Gradients", []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )

            explainer_dag_node = get_dag_node_for_id(
                call_info_singleton_alibi.mlinspect_explainer_node_id
            )
            add_dag_node(
                dag_node, [explainer_dag_node, test_data_node], backend_result
            )
            return result

        if not call_info_singleton_alibi.param_search_active:
            new_result = execute_patched_func_indirect_allowed(
                execute_inspections
            )
        else:
            original = gorilla.get_original_attribute(
                alibi.explainers.IntegratedGradients, "explain"
            )
            new_result = original(self, **args)
        return new_result

    @gorilla.patch(
        alibi.explainers.ALE,
        name="__init__",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched__init___ale(
        self,
        predictor: Callable[[np.ndarray], np.ndarray],
        feature_names: List[str] | None = None,
        target_names: List[str] | None = None,
        check_feature_resolution: bool = True,
        low_resolution_threshold: int = 10,
        extrapolate_constant: bool = True,
        extrapolate_constant_perc: float = 10.0,
        extrapolate_constant_min: float = 0.1,
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str | None = None,
        mlinspect_fit_transform_active: bool = False,
    ) -> Any:
        """Patch for ('alibi.explainers', 'ALE')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(
            alibi.explainers.ALE, "__init__"
        )

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = (
            mlinspect_optional_code_reference
        )
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_non_data_func_args = {
            "predictor": predictor,
            "feature_names": feature_names,
            "target_names": target_names,
            "check_feature_resolution": check_feature_resolution,
            "low_resolution_threshold": low_resolution_threshold,
            "extrapolate_constant": extrapolate_constant,
            "extrapolate_constant_perc": extrapolate_constant_perc,
            "extrapolate_constant_min": extrapolate_constant_min,
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
            call_info_singleton_alibi.mlinspect_explainer_node_id = (
                singleton.get_next_op_id()
            )

            function_info = FunctionInfo("alibi.explainers.ALE", "__init__")

            operator_context = OperatorContext(
                OperatorType.CREATE_EXPLAINER, function_info
            )
            input_infos = ExplainabilityBackend().before_call(
                operator_context, []
            )
            result = original(self, **self.mlinspect_non_data_func_args)
            backend_result = ExplainabilityBackend().after_call(
                operator_context,
                input_infos,
                result,
                self.mlinspect_non_data_func_args,
            )

            dag_node = DagNode(
                call_info_singleton_alibi.mlinspect_explainer_node_id,
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails("Alibi Explainer", []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            parent_nodes = []
            if call_info_singleton_alibi.parent_nodes_ale:
                parent_nodes.extend(call_info_singleton_alibi.parent_nodes_ale)
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
        alibi.explainers.ALE,
        name="explain",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched_explain_ale(
        self,
        X: np.ndarray,
        features: List[int] | None = None,
        min_bin_points: int = 4,
        grid_points: Dict[int, np.ndarray] | None = None,
    ) -> Any:
        """Patch for ('alibi.explainers.ALE', 'explain')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            alibi.explainers.ALE, "explain"
        )
        args = {
            "X": X,
            "features": features,
            "min_bin_points": min_bin_points,
            "grid_points": grid_points,
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
            function_info = FunctionInfo("alibi.explainers.ALE", "explain")
            # Test data
            data_backend_result, test_data_node, _ = add_test_data_dag_node(
                X,
                function_info,
                lineno,
                optional_code_reference,
                optional_source_code,
                caller_filename,
            )

            operator_context = OperatorContext(
                OperatorType.EXPLAINABILITY, function_info
            )
            input_dfs = [data_backend_result.annotated_dfobject]
            input_infos = ExplainabilityBackend().before_call(
                operator_context, input_dfs
            )

            result = original(self, **args)
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
                DagNodeDetails("ALE", []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )

            explainer_dag_node = get_dag_node_for_id(
                call_info_singleton_alibi.mlinspect_explainer_node_id
            )
            add_dag_node(
                dag_node, [explainer_dag_node, test_data_node], backend_result
            )
            return result

        if not call_info_singleton_alibi.param_search_active:
            new_result = execute_patched_func_indirect_allowed(
                execute_inspections
            )
        else:
            original = gorilla.get_original_attribute(
                alibi.explainers.ALE, "explain"
            )
            new_result = original(self, **args)
        return new_result
