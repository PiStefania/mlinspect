import gorilla
from lime import lime_tabular
from sklearn.inspection import PartialDependenceDisplay

from mlinspect.backends._shap_backend import ShapBackend

from mlinspect import FunctionInfo, OperatorContext, OperatorType, DagNode, BasicCodeLocation, DagNodeDetails
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._monkey_patching_utils import execute_patched_func_no_op_id, add_test_data_dag_node, \
    get_optional_code_info_or_none, get_dag_node_for_id, add_dag_node, execute_patched_func_indirect_allowed


class SklearnInspectionCallInfo:
    param_search_active: bool = False
    mlinspect_explainer_node_id = None
    parent_nodes = None
    actual_explainer_input = None

call_info_singleton_sklearn_inspection = SklearnInspectionCallInfo()

@gorilla.patches(PartialDependenceDisplay)
class SklearnInspectionPatching:
    """ Patches for sklearn.inspection """

    # pylint: disable=too-few-public-methods

    @gorilla.filter(False)
    def __init__(self, pd_results, features, feature_names, target_idx, deciles, kind="average",
                        subsample=1000, random_state=None, is_categorical=None, mlinspect_caller_filename=None,
                        mlinspect_lineno=None, mlinspect_optional_code_reference=None,
                        mlinspect_optional_source_code=None,mlinspect_fit_transform_active=False):
        """ Patch for ('sklearn.inspection', 'PartialDependenceDisplay') """
        ...


    @gorilla.settings(allow_hit=True)
    @classmethod
    def from_estimator(cls, *, estimator, X, features, sample_weight=None, categorical_features=None, feature_names=None,
        target=None, response_method="auto", n_cols=3, grid_resolution=100, percentiles=(0.05, 0.95), method="auto",
        n_jobs=None, verbose=0, line_kw=None, ice_lines_kw=None, pd_line_kw=None, contour_kw=None, ax=None,
        kind="average", centered=False, subsample=1000, random_state=None, mlinspect_caller_filename=None,
        mlinspect_lineno=None, mlinspect_optional_code_reference=None,
        mlinspect_optional_source_code=None, mlinspect_fit_transform_active=False):
        """ Patch for ('PartialDependenceDisplay', 'from_estimator') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(PartialDependenceDisplay, 'from_estimator')
        cls.mlinspect_caller_filename = mlinspect_caller_filename
        cls.mlinspect_lineno = mlinspect_lineno
        cls.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        cls.mlinspect_optional_source_code = mlinspect_optional_source_code
        cls.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        cls.mlinspect_non_data_func_args = {"estimator": estimator, "X": X,
                                             "features": features, "sample_weight": sample_weight,
                                             "categorical_features": categorical_features,
                                             "feature_names": feature_names, "target": target,
                                             "response_method": response_method, "n_cols": n_cols,
                                             "grid_resolution": grid_resolution, "percentiles": percentiles,
                                             "method": method, "n_jobs": n_jobs,
                                             "verbose": verbose, "line_kw": line_kw, "ice_lines_kw": ice_lines_kw,
                                             "pd_line_kw": pd_line_kw, "contour_kw": contour_kw, "ax": ax, "kind": kind,
                                             "centered": centered, "subsample": subsample, "random_state": random_state
                                             }
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            cls.mlinspect_caller_filename = caller_filename
            cls.mlinspect_lineno = lineno
            cls.mlinspect_optional_code_reference = optional_code_reference
            cls.mlinspect_optional_source_code = optional_source_code
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('PartialDependenceDisplay', 'from_estimator')
            # Test data
            data_backend_result, test_data_node, test_data_result = add_test_data_dag_node(X,
                                                                                           function_info,
                                                                                           lineno,
                                                                                           optional_code_reference,
                                                                                           optional_source_code,
                                                                                           caller_filename)

            operator_context = OperatorContext(OperatorType.EXPLAINABILITY, function_info)
            input_dfs = [data_backend_result.annotated_dfobject]
            input_infos = ShapBackend.before_call(operator_context, input_dfs)
            result = original(**cls.mlinspect_non_data_func_args)
            backend_result = ShapBackend.after_call(operator_context,
                                                    input_infos,
                                                    result,
                                                    cls.mlinspect_non_data_func_args)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("PDP" if kind == "average" else "ICE", []),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))

            add_dag_node(dag_node, [test_data_node, call_info_singleton_sklearn_inspection.parent_nodes[0]],
                         backend_result)
            return result

        if not call_info_singleton_sklearn_inspection.param_search_active:
            new_result = execute_patched_func_indirect_allowed(execute_inspections)
        else:
            original = gorilla.get_original_attribute(PartialDependenceDisplay, 'from_estimator')
            new_result = original(**cls.mlinspect_non_data_func_args)
        return new_result
