from typing import Any, List

import gorilla
from numpy.random import RandomState
from sklearn.base import BaseEstimator
from sklearn.inspection import PartialDependenceDisplay

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
    get_optional_code_info_or_none,
)


class SklearnInspectionCallInfo:
    param_search_active: bool = False
    mlinspect_explainer_node_id: int = -1
    parent_nodes: List[DagNode] | None = None
    actual_explainer_input: Any | None = None


call_info_singleton_sklearn_inspection = SklearnInspectionCallInfo()


@gorilla.patches(PartialDependenceDisplay)
class SklearnInspectionPatching:
    """Patches for sklearn.inspection"""

    # pylint: disable=too-few-public-methods

    @gorilla.filter(False)
    def __init__(  # type: ignore[no-untyped-def]
        self,
        pd_results: Any,
        features,
        feature_names,
        target_idx,
        deciles,
        kind="average",
        subsample=1000,
        random_state=None,
        is_categorical=None,
        mlinspect_caller_filename=None,
        mlinspect_lineno=None,
        mlinspect_optional_code_reference=None,
        mlinspect_optional_source_code=None,
        mlinspect_fit_transform_active=False,
    ) -> None:
        """Patch for ('sklearn.inspection', 'PartialDependenceDisplay')"""
        ...

    @gorilla.settings(allow_hit=True)
    @classmethod
    def from_estimator(
        cls,
        *,
        estimator: BaseEstimator,
        X: Any,
        features: list,
        sample_weight: Any | None = None,
        categorical_features: Any | None = None,
        feature_names: Any | None = None,
        target: int | None = None,
        response_method: str = "auto",
        n_cols: int = 3,
        grid_resolution: int = 100,
        percentiles: tuple[float, float] = (0.05, 0.95),
        method: str = "auto",
        n_jobs: int | None = None,
        verbose: int = 0,
        line_kw: dict | None = None,
        ice_lines_kw: dict | None = None,
        pd_line_kw: dict | None = None,
        contour_kw: dict | None = None,
        ax: Any | None = None,
        kind: str = "average",
        centered: bool = False,
        subsample: float | int | None = 1000,
        random_state: int | RandomState | None = None,
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str | None = None,
        mlinspect_fit_transform_active: bool = False
    ) -> Any:
        """Patch for ('PartialDependenceDisplay', 'from_estimator')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            PartialDependenceDisplay, "from_estimator"
        )
        cls.mlinspect_caller_filename = mlinspect_caller_filename  # type: ignore[attr-defined]
        cls.mlinspect_lineno = mlinspect_lineno  # type: ignore[attr-defined]
        cls.mlinspect_optional_code_reference = (  # type: ignore[attr-defined]
            mlinspect_optional_code_reference
        )
        cls.mlinspect_optional_source_code = mlinspect_optional_source_code  # type: ignore[attr-defined]
        cls.mlinspect_fit_transform_active = mlinspect_fit_transform_active  # type: ignore[attr-defined]
        cls.mlinspect_non_data_func_args = {  # type: ignore[attr-defined]
            "estimator": estimator,
            "X": X,
            "features": features,
            "sample_weight": sample_weight,
            "categorical_features": categorical_features,
            "feature_names": feature_names,
            "target": target,
            "response_method": response_method,
            "n_cols": n_cols,
            "grid_resolution": grid_resolution,
            "percentiles": percentiles,
            "method": method,
            "n_jobs": n_jobs,
            "verbose": verbose,
            "line_kw": line_kw,
            "ice_lines_kw": ice_lines_kw,
            "pd_line_kw": pd_line_kw,
            "contour_kw": contour_kw,
            "ax": ax,
            "kind": kind,
            "centered": centered,
            "subsample": subsample,
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
            cls.mlinspect_caller_filename = caller_filename  # type: ignore[attr-defined]
            cls.mlinspect_lineno = lineno  # type: ignore[attr-defined]
            cls.mlinspect_optional_code_reference = optional_code_reference  # type: ignore[attr-defined]
            cls.mlinspect_optional_source_code = optional_source_code  # type: ignore[attr-defined]
            # pylint: disable=too-many-locals
            function_info = FunctionInfo(
                "PartialDependenceDisplay", "from_estimator"
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
            result = original(**cls.mlinspect_non_data_func_args)  # type: ignore[attr-defined]
            backend_result = ExplainabilityBackend().after_call(
                operator_context,
                input_infos,
                result,
                cls.mlinspect_non_data_func_args,  # type: ignore[attr-defined]
            )
            dag_node = DagNode(
                singleton.get_next_op_id(),
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails("PDP" if kind == "average" else "ICE", []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            parent_nodes = [test_data_node]
            if call_info_singleton_sklearn_inspection.parent_nodes:
                parent_nodes.extend(
                    call_info_singleton_sklearn_inspection.parent_nodes
                )

            add_dag_node(
                dag_node,
                parent_nodes,
                backend_result,
            )
            return result

        if not call_info_singleton_sklearn_inspection.param_search_active:
            new_result = execute_patched_func_indirect_allowed(
                execute_inspections
            )
        else:
            original = gorilla.get_original_attribute(
                PartialDependenceDisplay, "from_estimator"
            )
            new_result = original(**cls.mlinspect_non_data_func_args)  # type: ignore[attr-defined]
        return new_result
