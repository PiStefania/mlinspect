import gorilla
import shap
from shap.utils._legacy import IdentityLink


from mlinspect.monkeypatching._monkey_patching_utils import execute_patched_func_no_op_id


class ShapPatching:
    """ Patches for pandas """

    # pylint: disable=too-few-public-methods

    @gorilla.patch(shap.KernelExplainer, name="__init__", settings=gorilla.Settings(allow_hit=True))
    def patched__init__(self, model, data, feature_names=None, link=IdentityLink(), mlinspect_caller_filename=None,
                        mlinspect_lineno=None, mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False):
        """ Patch for ('sklearn.feature_extraction.text', 'HashingVectorizer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        print("PLEASSEEEEEEEE!")
        original = gorilla.get_original_attribute(shap.KernelExplainer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active

        self.mlinspect_non_data_func_args = { 'model': model, 'data': data, 'feature_names': feature_names, 'link': link}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)