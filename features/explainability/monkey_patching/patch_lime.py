import gorilla
import numpy as np
from lime import lime_tabular
import shap
from mlinspect.backends._shap_backend import ShapBackend

from mlinspect import FunctionInfo, OperatorContext, OperatorType, DagNode, BasicCodeLocation, DagNodeDetails
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._mlinspect_ndarray import MlinspectNdarray
from mlinspect.monkeypatching._monkey_patching_utils import execute_patched_func_no_op_id, add_test_data_dag_node, \
    get_optional_code_info_or_none, get_dag_node_for_id, add_dag_node, execute_patched_func_indirect_allowed


class LimeCallInfo:
    param_search_active: bool = False
    mlinspect_explainer_node_id = None
    parent_nodes = None
    actual_explainer_input = None


call_info_singleton_lime = LimeCallInfo()


class LimePatching:
    """ Patches for pandas """

    # pylint: disable=too-few-public-methods

    @gorilla.patch(lime_tabular.LimeTabularExplainer, name="__init__", settings=gorilla.Settings(allow_hit=True))
    def patched__init__(self, training_data, mode="classification", training_labels=None, feature_names=None,
                        categorical_features=None, categorical_names=None, kernel_width=None, kernel=None, verbose=False,
                        class_names=None, feature_selection='auto', discretize_continuous=True, discretizer='quartile', sample_around_instance=False,
                        random_state=None, training_data_stats=None, mlinspect_caller_filename=None,
                        mlinspect_lineno=None, mlinspect_optional_code_reference=None,
                        mlinspect_optional_source_code=None,mlinspect_fit_transform_active=False):
        """ Patch for ('lime.lime_tabular', 'LimeTabularExplainer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(lime_tabular.LimeTabularExplainer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_non_data_func_args = {"training_data": training_data, "mode": mode, "training_labels": training_labels, "feature_names": feature_names,
                        "categorical_features": categorical_features, "categorical_names": categorical_names, "kernel_width": kernel_width, "kernel":kernel, "verbose": verbose,
                        "class_names": class_names, "feature_selection": feature_selection, "discretize_continuous": discretize_continuous, "discretizer": discretizer, "sample_around_instance": sample_around_instance,
                        "random_state": random_state, "training_data_stats": training_data_stats}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code
            call_info_singleton_lime.mlinspect_explainer_node_id = singleton.get_next_op_id()

            function_info = FunctionInfo('lime.lime_tabular.LimeTabularExplainer', '__init__')
            data_backend_result, test_data_node, test_data_result = add_test_data_dag_node(training_data,
                                                                                           function_info,
                                                                                           lineno,
                                                                                           optional_code_reference,
                                                                                           optional_source_code,
                                                                                           caller_filename)

            operator_context = OperatorContext(OperatorType.CREATE_EXPLAINER, function_info)

            input_dfs = [data_backend_result.annotated_dfobject]
            input_infos = ShapBackend.before_call(operator_context, input_dfs)
            result = original(self, **self.mlinspect_non_data_func_args)
            backend_result = ShapBackend.after_call(operator_context,
                                                    input_infos,
                                                    result,
                                                    self.mlinspect_non_data_func_args)


            dag_node = DagNode(call_info_singleton_lime.mlinspect_explainer_node_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("Neural Network", []),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [test_data_node], backend_result)

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.patch(lime_tabular.LimeTabularExplainer, name='explain_instance', settings=gorilla.Settings(allow_hit=True))
    def patched_explain_instance(self, *args, **kwargs):
        """ Patch for ('lime_tabular.LimeTabularExplainer', 'explain_instance') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(lime_tabular.LimeTabularExplainer, 'explain_instance')

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('lime.lime_tabular.LimeTabularExplainer', 'explain_instance')
            # Test data
            data_backend_result, test_data_node, test_data_result = add_test_data_dag_node(args[0],
                                                                                           function_info,
                                                                                           lineno,
                                                                                           optional_code_reference,
                                                                                           optional_source_code,
                                                                                           caller_filename)

            operator_context = OperatorContext(OperatorType.EXPLAINABILITY, function_info)
            input_dfs = [data_backend_result.annotated_dfobject]
            input_infos = ShapBackend.before_call(operator_context, input_dfs)
            call_info_singleton_lime.actual_explainer_input = test_data_result
            result = original(self, *args, **kwargs)
            backend_result = ShapBackend.after_call(operator_context,
                                                    input_infos,
                                                    result,
                                                    self.mlinspect_non_data_func_args)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("LIME", []),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))

            explainer_dag_node = get_dag_node_for_id(call_info_singleton_lime.mlinspect_explainer_node_id)
            add_dag_node(dag_node, [explainer_dag_node, test_data_node, call_info_singleton_lime.parent_nodes[0]],
                         backend_result)
            return result

        if not call_info_singleton_lime.param_search_active:
            new_result = execute_patched_func_indirect_allowed(execute_inspections)
        else:
            original = gorilla.get_original_attribute(lime_tabular.LimeTabularExplainer, 'explain_instance')
            new_result = original(self, *args, **kwargs)
        return new_result
