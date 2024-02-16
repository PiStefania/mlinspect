import dalex
import gorilla
import numpy as np

from ..dale import dale

from mlinspect import FunctionInfo, OperatorContext, OperatorType, DagNode, BasicCodeLocation, DagNodeDetails
from mlinspect.backends._shap_backend import ShapBackend
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._monkey_patching_utils import get_optional_code_info_or_none, add_dag_node, \
    execute_patched_func_no_op_id, add_test_data_dag_node, get_dag_node_for_id, execute_patched_func_indirect_allowed, add_test_label_node


class DalexCallInfo:
    param_search_active: bool = False
    mlinspect_explainer_node_id = None
    parent_nodes = None


call_info_singleton_dalex = DalexCallInfo()

class DalexPatching:
    """ Patches for DALEX """

    # pylint: disable=too-few-public-methods

    @gorilla.patch(dalex.Explainer, name="__init__", settings=gorilla.Settings(allow_hit=True))
    def patched__init__(self, model, data=None, y=None, predict_function=None, residual_function=None,  weights=None,
                 label=None, model_class=None, verbose=True, precalculate=True, model_type=None, model_info=None,
                        mlinspect_caller_filename=None,
                        mlinspect_lineno=None, mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False):
        """ Patch for ('dalex', 'Explainer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(dalex.Explainer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_non_data_func_args = {"model": model, "data": data.view(np.ndarray), "y": y.view(np.ndarray), "predict_function": predict_function,
                                             "residual_function": residual_function, "weights": weights, "label": label,
                                             "model_class": model_class, "verbose": verbose, "precalculate": precalculate,
                                             "model_type": model_type, "model_info": model_info}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code
            call_info_singleton_dalex.mlinspect_explainer_node_id = singleton.get_next_op_id()

            function_info = FunctionInfo('dalex.Explainer', '__init__')

            operator_context = OperatorContext(OperatorType.CREATE_EXPLAINER, function_info)

            # Test data
            data_backend_result, test_data_node, test_data_result = add_test_data_dag_node(data,
                                                                                           function_info,
                                                                                           lineno,
                                                                                           optional_code_reference,
                                                                                           optional_source_code,
                                                                                           caller_filename)
            # Test labels
            label_backend_result, test_labels_node, test_labels_result = add_test_label_node(y,
                                                                                             caller_filename,
                                                                                             function_info,
                                                                                             lineno,
                                                                                             optional_code_reference,
                                                                                             optional_source_code)

            input_infos = ShapBackend.before_call(operator_context, [])
            result = original(self, **self.mlinspect_non_data_func_args)
            backend_result = ShapBackend.after_call(operator_context,
                                                    input_infos,
                                                    result,
                                                    self.mlinspect_non_data_func_args)

            dag_node = DagNode(call_info_singleton_dalex.mlinspect_explainer_node_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("Neural Network", []),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, [call_info_singleton_dalex.parent_nodes[0], test_data_node, test_labels_node], backend_result)

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.patch(dalex.Explainer, name='model_parts', settings=gorilla.Settings(allow_hit=True))
    def patched_model_parts(self, loss_function=None, type=('variable_importance', 'ratio', 'difference', 'shap_wrapper'),
                N=1000, B=10, variables=None, variable_groups=None, keep_raw_permutations=True, label=None,
                processes=1, random_state=None, **kwargs):
        """ Patch for ('dalex.Explainer', 'model_parts') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(dalex.Explainer, 'model_parts')
        args = {"loss_function": loss_function, "type": type, "N": N, "B":B, "variables": variables, "variable_groups": variable_groups,
                "keep_raw_permutations": keep_raw_permutations, "label": label, "processes": processes, "random_state": random_state}
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('dalex.Explainer', 'model_parts')
            operator_context = OperatorContext(OperatorType.EXPLAINABILITY, function_info)
            input_dfs = []
            input_infos = ShapBackend.before_call(operator_context, input_dfs)
            result = original(self, **args, **kwargs)
            backend_result = ShapBackend.after_call(operator_context,
                                                    input_infos,
                                                                 result,
                                                                 self.mlinspect_non_data_func_args)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("DALEX", []),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))

            explainer_dag_node = get_dag_node_for_id(call_info_singleton_dalex.mlinspect_explainer_node_id)
            add_dag_node(dag_node, [explainer_dag_node],
                         backend_result)
            return result
        if not call_info_singleton_dalex.param_search_active:
            new_result = execute_patched_func_indirect_allowed(execute_inspections)
        else:
            original = gorilla.get_original_attribute(dalex.Explainer, 'model_parts')
            new_result = original(self, **args)
        return new_result

    @gorilla.patch(dalex.Explainer, name='predict_parts', settings=gorilla.Settings(allow_hit=True))
    def patched_predict_parts(self, new_observation,
                type=('break_down_interactions', 'break_down', 'shap', 'shap_wrapper'), order=None,
                interaction_preference=1, path="average", N=None, B=25, keep_distributions=False,
                label=None, processes=1, random_state=None, **kwargs):
        """ Patch for ('dalex.Explainer', 'predict_parts') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(dalex.Explainer, 'predict_parts')
        args = {"new_observation": new_observation, "type": type, "N": N, "B": B, "order": order,
                "interaction_preference": interaction_preference,
                "path": path, "label": label, "processes": processes,
                "random_state": random_state, "keep_distributions": keep_distributions}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('dalex.Explainer', 'predict_parts')
            operator_context = OperatorContext(OperatorType.EXPLAINABILITY, function_info)
            # Test data
            data_backend_result, test_data_node, test_data_result = add_test_data_dag_node(new_observation,
                                                                                           function_info,
                                                                                           lineno,
                                                                                           optional_code_reference,
                                                                                           optional_source_code,
                                                                                           caller_filename)
            input_dfs = []
            input_infos = ShapBackend.before_call(operator_context, input_dfs)
            print(new_observation.shape)
            result = original(self, **args, **kwargs)
            backend_result = ShapBackend.after_call(operator_context,
                                                    input_infos,
                                                    result,
                                                    self.mlinspect_non_data_func_args)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("DALEX", []),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))

            explainer_dag_node = get_dag_node_for_id(call_info_singleton_dalex.mlinspect_explainer_node_id)
            add_dag_node(dag_node, [explainer_dag_node, test_data_node],
                         backend_result)
            return result

        if not call_info_singleton_dalex.param_search_active:
            new_result = execute_patched_func_indirect_allowed(execute_inspections)
        else:
            original = gorilla.get_original_attribute(dalex.Explainer, 'predict_parts')
            new_result = original(self, **args)
        return new_result