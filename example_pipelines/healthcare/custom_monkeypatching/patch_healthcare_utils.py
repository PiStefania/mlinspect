"""
Monkey patching for healthcare_utils
"""

from typing import Any, Callable

import gorilla

from example_pipelines.healthcare import healthcare_utils

from mlinspect import CodeReference
from mlinspect.backends._sklearn_backend import SklearnBackend
from mlinspect.inspections._inspection_input import (
    FunctionInfo,
    OperatorContext,
    OperatorType,
)
from mlinspect.instrumentation._dag_node import (
    BasicCodeLocation,
    DagNode,
    DagNodeDetails,
)
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._mlinspect_ndarray import MlinspectNdarray
from mlinspect.monkeypatching._monkey_patching_utils import (
    add_dag_node,
    execute_patched_func_no_op_id,
    get_input_info,
    get_optional_code_info_or_none,
)


class SklearnMyW2VTransformerPatching:
    """Patches for healthcare_utils.MyW2VTransformer"""

    # pylint: disable=too-few-public-methods

    @gorilla.patch(
        healthcare_utils.W2VTransformer,
        name="__init__",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched__init__(
        self,
        *,
        vector_size: int = 100,
        alpha: float = 0.025,
        window: int = 5,
        min_count: int = 5,
        max_vocab_size: int | None = None,
        sample: float = 1e-3,
        seed: int = 1,
        workers: int = 3,
        min_alpha: float = 0.0001,
        sg: int = 0,
        hs: int = 0,
        negative: int = 5,
        cbow_mean: int = 1,
        hashfxn: Callable = hash,
        epochs: int = 5,
        null_word: int = 0,
        trim_rule: Callable | None = None,
        sorted_vocab: int = 1,
        batch_words: int = 10000,
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str | None = None,
        mlinspect_fit_transform_active: bool = False
    ) -> Any:
        """Patch for ('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals, redefined-builtin,
        # pylint: disable=invalid-name
        original = gorilla.get_original_attribute(
            healthcare_utils.W2VTransformer, "__init__"
        )

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = (
            mlinspect_optional_code_reference
        )
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active

        self.mlinspect_non_data_func_args = {
            "vector_size": vector_size,
            "alpha": alpha,
            "window": window,
            "min_count": min_count,
            "max_vocab_size": max_vocab_size,
            "sample": sample,
            "seed": seed,
            "workers": workers,
            "min_alpha": min_alpha,
            "sg": sg,
            "hs": hs,
            "negative": negative,
            "cbow_mean": cbow_mean,
            "epochs": epochs,
            "null_word": null_word,
            "trim_rule": trim_rule,
            "sorted_vocab": sorted_vocab,
            "batch_words": batch_words,
        }

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> None:
            """Execute inspections, add DAG node"""
            original(
                self, hashfxn=hashfxn, **self.mlinspect_non_data_func_args
            )

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(
            original,
            execute_inspections,
            self,
            hashfxn=hashfxn,
            **self.mlinspect_non_data_func_args
        )

    @gorilla.patch(
        healthcare_utils.MyW2VTransformer,
        name="fit_transform",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched_fit_transform(
        self, *args: Any, **kwargs: Any
    ) -> MlinspectNdarray:
        """Patch for ('example_pipelines.healthcare.healthcare_utils.MyW2VTransformer', 'fit_transform')"""
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = (
            True  # pylint: disable=attribute-defined-outside-init
        )
        original = gorilla.get_original_attribute(
            healthcare_utils.MyW2VTransformer, "fit_transform"
        )
        function_info = FunctionInfo(
            "example_pipelines.healthcare.healthcare_utils", "MyW2VTransformer"
        )
        input_info = get_input_info(
            args[0],
            self.mlinspect_caller_filename,
            self.mlinspect_lineno,
            function_info,
            self.mlinspect_optional_code_reference,
            self.mlinspect_optional_source_code,
        )

        operator_context = OperatorContext(
            OperatorType.TRANSFORMER, function_info
        )
        input_infos = SklearnBackend().before_call(
            operator_context, [input_info.annotated_dfobject]
        )
        result = original(
            self, input_infos[0].result_data, *args[1:], **kwargs
        )
        backend_result = SklearnBackend().after_call(
            operator_context,
            input_infos,
            result,
            self.mlinspect_non_data_func_args,
        )
        new_return_value = backend_result.annotated_dfobject.result_data
        assert isinstance(new_return_value, MlinspectNdarray)
        dag_node = DagNode(
            singleton.get_next_op_id(),
            BasicCodeLocation(
                self.mlinspect_caller_filename, self.mlinspect_lineno
            ),
            operator_context,
            DagNodeDetails("Word2Vec: fit_transform", ["array"]),
            get_optional_code_info_or_none(
                self.mlinspect_optional_code_reference,
                self.mlinspect_optional_source_code,
            ),
        )
        add_dag_node(dag_node, [input_info.dag_node], backend_result)
        self.mlinspect_fit_transform_active = (
            False  # pylint: disable=attribute-defined-outside-init
        )
        return new_return_value

    @gorilla.patch(
        healthcare_utils.MyW2VTransformer,
        name="transform",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('example_pipelines.healthcare.healthcare_utils.MyW2VTransformer', 'transform')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            healthcare_utils.MyW2VTransformer, "transform"
        )
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo(
                "example_pipelines.healthcare.healthcare_utils",
                "MyW2VTransformer",
            )
            input_info = get_input_info(
                args[0],
                self.mlinspect_caller_filename,
                self.mlinspect_lineno,
                function_info,
                self.mlinspect_optional_code_reference,
                self.mlinspect_optional_source_code,
            )

            operator_context = OperatorContext(
                OperatorType.TRANSFORMER, function_info
            )
            input_infos = SklearnBackend().before_call(
                operator_context, [input_info.annotated_dfobject]
            )
            result = original(
                self, input_infos[0].result_data, *args[1:], **kwargs
            )
            backend_result = SklearnBackend().after_call(
                operator_context,
                input_infos,
                result,
                self.mlinspect_non_data_func_args,
            )
            new_return_value = backend_result.annotated_dfobject.result_data
            assert isinstance(new_return_value, MlinspectNdarray)
            dag_node = DagNode(
                singleton.get_next_op_id(),
                BasicCodeLocation(
                    self.mlinspect_caller_filename, self.mlinspect_lineno
                ),
                operator_context,
                DagNodeDetails("Word2Vec: transform", ["array"]),
                get_optional_code_info_or_none(
                    self.mlinspect_optional_code_reference,
                    self.mlinspect_optional_source_code,
                ),
            )
            add_dag_node(dag_node, [input_info.dag_node], backend_result)
        else:
            new_return_value = original(self, *args, **kwargs)
        return new_return_value
