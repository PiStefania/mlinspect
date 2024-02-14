import alibi
import gorilla
import numpy as np
import shap
from mlinspect.backends._shap_backend import ShapBackend
from shap.utils._legacy import IdentityLink

from mlinspect import FunctionInfo, OperatorContext, OperatorType, DagNode, BasicCodeLocation, DagNodeDetails
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._mlinspect_ndarray import MlinspectNdarray
from mlinspect.monkeypatching._monkey_patching_utils import execute_patched_func_no_op_id, add_test_data_dag_node, \
    get_optional_code_info_or_none, get_dag_node_for_id, add_dag_node, execute_patched_func_indirect_allowed


class AlibiCallInfo:
    param_search_active: bool = False
    mlinspect_explainer_node_id = None
    parent_nodes = None
    actual_explainer_input = None


call_info_singleton_alibi = AlibiCallInfo()

class AlibiPatching:
    """ Patches for pandas """

    # pylint: disable=too-few-public-methods

    @gorilla.patch(alibi.explainers.IntegratedGradients, name="__init__", settings=gorilla.Settings(allow_hit=True))
    def patched__init__(self, model, layer = None, target_fn = None, method = "gausslegendre", n_steps = 50,
                 internal_batch_size = 100, mlinspect_caller_filename=None,
                mlinspect_lineno=None, mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                mlinspect_fit_transform_active=False):
        """ Patch for ('alibi.explainers', 'IntegratedGradients') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(alibi.explainers.IntegratedGradients, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_non_data_func_args = {'model': model, 'layer': layer, 'target_fn': target_fn, 'method': method, "n_steps": n_steps,
                                             "internal_batch_size": internal_batch_size, }

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code
            call_info_singleton_alibi.mlinspect_explainer_node_id = singleton.get_next_op_id()

            function_info = FunctionInfo('alibi.explainers.IntegratedGradients', '__init__')

            operator_context = OperatorContext(OperatorType.CREATE_EXPLAINER, function_info)
            input_infos = ShapBackend.before_call(operator_context, [])
            result = original(self, **self.mlinspect_non_data_func_args)
            backend_result = ShapBackend.after_call(operator_context,
                                                    input_infos,
                                                    result,
                                                    self.mlinspect_non_data_func_args)

            dag_node = DagNode(call_info_singleton_alibi.mlinspect_explainer_node_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("Neural Network", []),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, call_info_singleton_alibi.parent_nodes, backend_result)

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.patch(alibi.explainers.IntegratedGradients, name='explain', settings=gorilla.Settings(allow_hit=True))
    def patched_explain(self, X, forward_kwargs=None, baselines=None, target=None,
                attribute_to_layer_inputs=False):
        """ Patch for ('alibi.explainers.IntegratedGradients', 'explain') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(alibi.explainers.IntegratedGradients, 'explain')
        args={'X': X, 'forward_kwargs': forward_kwargs, 'baselines': baselines, 'target': target, "attribute_to_layer_inputs": attribute_to_layer_inputs}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('alibi.explainers.IntegratedGradients', 'explain')
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

            result = original(self, **args)
            backend_result = ShapBackend.after_call(operator_context,
                                                    input_infos,
                                                    result,
                                                    self.mlinspect_non_data_func_args)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("Shapley Values", []),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))

            explainer_dag_node = get_dag_node_for_id(call_info_singleton_alibi.mlinspect_explainer_node_id)
            add_dag_node(dag_node, [explainer_dag_node, test_data_node],
                         backend_result)
            return result

        if not call_info_singleton_alibi.param_search_active:
            new_result = execute_patched_func_indirect_allowed(execute_inspections)
        else:
            original = gorilla.get_original_attribute(alibi.explainers.IntegratedGradients, 'explain')
            new_result = original(self, **args)
        return new_result