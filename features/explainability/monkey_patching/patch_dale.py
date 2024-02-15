import gorilla
import numpy as np

from ..dale import dale

from mlinspect import FunctionInfo, OperatorContext, OperatorType, DagNode, BasicCodeLocation, DagNodeDetails
from mlinspect.backends._shap_backend import ShapBackend
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._monkey_patching_utils import get_optional_code_info_or_none, add_dag_node, \
    execute_patched_func_no_op_id, add_test_data_dag_node, get_dag_node_for_id, execute_patched_func_indirect_allowed


class DaleCallInfo:
    param_search_active: bool = False
    mlinspect_explainer_node_id = None
    parent_nodes = None


call_info_singleton_dale = DaleCallInfo()

class DalePatching:
    """ Patches for DALE """

    # pylint: disable=too-few-public-methods

    @gorilla.patch(dale.DALE, name="__init__", settings=gorilla.Settings(allow_hit=True))
    def patched__init__(self, data, model, model_jac=None,
                        mlinspect_caller_filename=None,
                        mlinspect_lineno=None, mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False):
        """ Patch for ('dale', 'DALE') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(dale.DALE, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_non_data_func_args = { 'data': data.view(np.ndarray), 'model': model, 'model_jac': model_jac}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code
            call_info_singleton_dale.mlinspect_explainer_node_id = singleton.get_next_op_id()

            function_info = FunctionInfo('dale.DALE', '__init__')

            operator_context = OperatorContext(OperatorType.CREATE_EXPLAINER, function_info)
            input_infos = ShapBackend.before_call(operator_context, [])
            result = original(self, **self.mlinspect_non_data_func_args)
            backend_result = ShapBackend.after_call(operator_context,
                                                    input_infos,
                                                    result,
                                                    self.mlinspect_non_data_func_args)

            dag_node = DagNode(call_info_singleton_dale.mlinspect_explainer_node_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("Neural Network", []),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            add_dag_node(dag_node, call_info_singleton_dale.parent_nodes, backend_result)

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.patch(dale.DALE, name='eval', settings=gorilla.Settings(allow_hit=True))
    def patched_shap_values(self, x: np.ndarray, s: int):
        """ Patch for ('dale.DALE', 'eval') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(dale.DALE, 'eval')
        args = {'x': x, 's': s}
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('dale.DALE', 'eval')
            # Test data
            data_backend_result, test_data_node, test_data_result = add_test_data_dag_node(x,
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
                               DagNodeDetails("DALE", []),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))

            explainer_dag_node = get_dag_node_for_id(call_info_singleton_dale.mlinspect_explainer_node_id)
            add_dag_node(dag_node, [explainer_dag_node, test_data_node],
                         backend_result)
            return result
        if not call_info_singleton_dale.param_search_active:
            new_result = execute_patched_func_indirect_allowed(execute_inspections)
        else:
            original = gorilla.get_original_attribute(dale.DALE, 'eval')
            new_result = original(self, **args)
        return new_result