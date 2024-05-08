"""
Monkey patching for sklearn
"""

from typing import Any, Callable, Dict, List, Union

import gorilla
import numpy
import pandas
import tensorflow as tf
from scikeras import (  # pylint: disable=reimported
    wrappers as keras_sklearn_external,
    wrappers as keras_sklearn_internal,
)
from sklearn import (
    compose,
    impute,
    linear_model,
    model_selection,
    preprocessing,
    tree,
)
from sklearn.feature_extraction import text
from sklearn.linear_model._stochastic_gradient import DEFAULT_EPSILON
from sklearn.metrics import accuracy_score

from features.explainability.monkey_patching.patch_alibi import (
    call_info_singleton_alibi,
)
from features.explainability.monkey_patching.patch_dale import (
    call_info_singleton_dale,
)
from features.explainability.monkey_patching.patch_dalex import (
    call_info_singleton_dalex,
)
from features.explainability.monkey_patching.patch_lime import (
    call_info_singleton_lime,
)
from features.explainability.monkey_patching.patch_shap import (
    call_info_singleton_shap,
)
from features.explainability.monkey_patching.patch_sklearn_inspection import (
    call_info_singleton_sklearn_inspection,
)

from mlinspect.backends._backend import BackendResult
from mlinspect.backends._sklearn_backend import SklearnBackend
from mlinspect.inspections._inspection_input import (
    FunctionInfo,
    OperatorContext,
    OperatorType,
)
from mlinspect.instrumentation._dag_node import (
    BasicCodeLocation,
    CodeReference,
    DagNode,
    DagNodeDetails,
)
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._mlinspect_ndarray import MlinspectNdarray
from mlinspect.monkeypatching._monkey_patching_utils import (
    add_dag_node,
    add_test_data_dag_node,
    add_test_label_node,
    add_train_data_node,
    add_train_label_node,
    execute_patched_func,
    execute_patched_func_indirect_allowed,
    execute_patched_func_no_op_id,
    get_dag_node_for_id,
    get_input_info,
    get_optional_code_info_or_none,
)

# pylint: disable=too-many-lines


@gorilla.patches(preprocessing)
class SklearnPreprocessingPatching:
    """Patches for sklearn"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("label_binarize")
    @gorilla.settings(allow_hit=True)
    def patched_label_binarize(  # pylint: disable=no-self-argument
        *args: Any, **kwargs: Any
    ) -> Any:
        """Patch for ('sklearn.preprocessing._label', 'label_binarize')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            preprocessing, "label_binarize"
        )

        def execute_inspections(
            op_id: int,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> Any:
            """Execute inspections, add DAG node"""
            # pylint: disable=too-many-locals
            function_info = FunctionInfo(
                "sklearn.preprocessing._label", "label_binarize"
            )
            input_info = get_input_info(
                args[0],
                caller_filename,
                lineno,
                function_info,
                optional_code_reference,
                optional_source_code,
            )

            operator_context = OperatorContext(
                OperatorType.PROJECTION_MODIFY, function_info
            )
            input_infos = SklearnBackend().before_call(
                operator_context, [input_info.annotated_dfobject]
            )
            result = original(input_infos[0].result_data, *args[1:], **kwargs)
            backend_result = SklearnBackend().after_call(
                operator_context, input_infos, result
            )
            new_return_value = backend_result.annotated_dfobject.result_data

            classes = kwargs["classes"]
            description = "label_binarize, classes: {}".format(classes)
            dag_node = DagNode(
                op_id,
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails(description, ["array"]),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            add_dag_node(dag_node, [input_info.dag_node], backend_result)

            return new_return_value

        return execute_patched_func(
            original, execute_inspections, *args, **kwargs
        )


@gorilla.patches(model_selection)
class SklearnModelSelectionPatching:
    """Patches for sklearn"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("train_test_split")
    @gorilla.settings(allow_hit=True)
    def patched_train_test_split(
        *args: Any, **kwargs: Any
    ) -> Any:  # pylint: disable=no-self-argument
        """Patch for ('sklearn.model_selection._split', 'train_test_split')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            model_selection, "train_test_split"
        )

        def execute_inspections(
            op_id: int,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> Any:
            """Execute inspections, add DAG node"""
            # pylint: disable=too-many-locals
            function_info = FunctionInfo(
                "sklearn.model_selection._split", "train_test_split"
            )
            input_info = get_input_info(
                args[0],
                caller_filename,
                lineno,
                function_info,
                optional_code_reference,
                optional_source_code,
            )

            operator_context = OperatorContext(
                OperatorType.TRAIN_TEST_SPLIT, function_info
            )
            input_infos = SklearnBackend().before_call(
                operator_context, [input_info.annotated_dfobject]
            )
            result = original(input_infos[0].result_data, *args[1:], **kwargs)
            backend_result = SklearnBackend().after_call(
                operator_context, input_infos, result
            )  # We ignore the test set for now
            train_backend_result = BackendResult(
                backend_result.annotated_dfobject,
                backend_result.dag_node_annotation,
            )
            test_backend_result = BackendResult(
                backend_result.optional_second_annotated_dfobject,
                backend_result.optional_second_dag_node_annotation,
            )

            description = "(Train Data)"
            columns = list(result[0].columns)
            dag_node = DagNode(
                op_id,
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails(description, columns),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            add_dag_node(dag_node, [input_info.dag_node], train_backend_result)

            description = "(Test Data)"
            columns = list(result[1].columns)
            dag_node = DagNode(
                singleton.get_next_op_id(),
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails(description, columns),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            if test_backend_result:
                add_dag_node(
                    dag_node, [input_info.dag_node], test_backend_result
                )

            new_return_value = (
                train_backend_result.annotated_dfobject.result_data,
                (
                    test_backend_result.annotated_dfobject.result_data
                    if test_backend_result
                    else None
                ),
            )

            return new_return_value

        return execute_patched_func(
            original, execute_inspections, *args, **kwargs
        )


class SklearnCallInfo:
    """Contains info like lineno from the current Transformer so indirect utility function calls can access it"""

    # pylint: disable=too-few-public-methods

    transformer_filename: str | None = None
    transformer_lineno: int | None = None
    transformer_function_info: FunctionInfo | None = None
    transformer_optional_code_reference: CodeReference | None = None
    transformer_optional_source_code: str | None = None
    column_transformer_active: bool = False
    param_search_active: bool = False


call_info_singleton = SklearnCallInfo()


@gorilla.patches(model_selection.GridSearchCV)
class SklearnGridSearchCVPatching:
    """Patches for sklearn GridSearchCV"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("__init__")
    @gorilla.settings(allow_hit=True)
    def patched__init__(
        self,
        estimator: Any,
        param_grid: Dict | List[Dict],
        *,
        scoring: Any | None = None,
        n_jobs: int | None = None,
        refit: bool = True,
        cv: Any | None = None,
        verbose: int = 0,
        pre_dispatch: int | str = "2*n_jobs",
        error_score: float = numpy.nan,
        return_train_score: bool = False,
    ) -> Any:
        """Patch for ('sklearn.compose.model_selection._search', 'GridSearchCV')"""
        # pylint: disable=no-method-argument,invalid-name
        original = gorilla.get_original_attribute(
            model_selection.GridSearchCV, "__init__"
        )

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> None:
            """Execute inspections, add DAG node"""
            # pylint: disable=attribute-defined-outside-init
            original(
                self,
                estimator,
                param_grid,
                scoring=scoring,
                n_jobs=n_jobs,
                refit=refit,
                cv=cv,
                verbose=verbose,
                pre_dispatch=pre_dispatch,
                error_score=error_score,
                return_train_score=return_train_score,
            )

            self.mlinspect_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_indirect_allowed(execute_inspections)

    @gorilla.name("_run_search")
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.compose.model_selection._search', 'GridSearchCV')"""
        # pylint: disable=no-method-argument
        supported_estimators = (
            tree.DecisionTreeClassifier,
            linear_model.SGDClassifier,
            linear_model.LogisticRegression,
            keras_sklearn_external.KerasClassifier,
        )
        if not isinstance(
            self.estimator, supported_estimators  # type: ignore[attr-defined]
        ):  # pylint: disable=no-member
            raise NotImplementedError(
                f"TODO: Estimator is an instance of "
                f"{type(self.estimator)}, "  # type: ignore[attr-defined]
                f"which is not supported yet!"
            )

        call_info_singleton.transformer_filename = self.mlinspect_filename
        call_info_singleton.transformer_lineno = self.mlinspect_lineno
        call_info_singleton.transformer_function_info = FunctionInfo(
            "sklearn.compose.model_selection._search.GridSearchCV",
            "_run_search",
        )
        call_info_singleton.transformer_optional_code_reference = (
            self.mlinspect_optional_code_reference
        )
        call_info_singleton.transformer_optional_source_code = (
            self.mlinspect_optional_source_code
        )

        call_info_singleton.param_search_active = True
        original = gorilla.get_original_attribute(
            model_selection.GridSearchCV, "_run_search"
        )
        result = original(self, *args, **kwargs)
        call_info_singleton.param_search_active = False

        return result


@gorilla.patches(compose.ColumnTransformer)
class SklearnComposePatching:
    """Patches for sklearn ColumnTransformer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("__init__")
    @gorilla.settings(allow_hit=True)
    def patched__init__(
        self,
        transformers: Any,
        *,
        remainder: str = "drop",
        sparse_threshold: float = 0.3,
        n_jobs: int | None = None,
        transformer_weights: dict | None = None,
        verbose: bool = False,
    ) -> Any:
        """Patch for ('sklearn.compose._column_transformer', 'ColumnTransformer')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            compose.ColumnTransformer, "__init__"
        )

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> None:
            """Execute inspections, add DAG node"""
            # pylint: disable=attribute-defined-outside-init
            original(
                self,
                transformers,
                remainder=remainder,
                sparse_threshold=sparse_threshold,
                n_jobs=n_jobs,
                transformer_weights=transformer_weights,
                verbose=verbose,
            )

            self.mlinspect_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_indirect_allowed(execute_inspections)

    @gorilla.name("fit_transform")
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.compose._column_transformer', 'ColumnTransformer')"""
        # pylint: disable=no-method-argument
        call_info_singleton.transformer_filename = self.mlinspect_filename
        call_info_singleton.transformer_lineno = self.mlinspect_lineno
        call_info_singleton.transformer_function_info = FunctionInfo(
            "sklearn.compose._column_transformer", "ColumnTransformer"
        )
        call_info_singleton.transformer_optional_code_reference = (
            self.mlinspect_optional_code_reference
        )
        call_info_singleton.transformer_optional_source_code = (
            self.mlinspect_optional_source_code
        )

        call_info_singleton.column_transformer_active = True
        original = gorilla.get_original_attribute(
            compose.ColumnTransformer, "fit_transform"
        )
        result = original(self, *args, **kwargs)
        call_info_singleton.column_transformer_active = False

        return result

    @gorilla.name("transform")
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.compose._column_transformer', 'ColumnTransformer')"""
        # pylint: disable=no-method-argument
        call_info_singleton.transformer_filename = self.mlinspect_filename
        call_info_singleton.transformer_lineno = self.mlinspect_lineno
        call_info_singleton.transformer_function_info = FunctionInfo(
            "sklearn.compose._column_transformer", "ColumnTransformer"
        )
        call_info_singleton.transformer_optional_code_reference = (
            self.mlinspect_optional_code_reference
        )
        call_info_singleton.transformer_optional_source_code = (
            self.mlinspect_optional_source_code
        )

        call_info_singleton.column_transformer_active = True
        original = gorilla.get_original_attribute(
            compose.ColumnTransformer, "transform"
        )
        result = original(self, *args, **kwargs)
        call_info_singleton.column_transformer_active = False

        return result

    @gorilla.name("_hstack")
    @gorilla.settings(allow_hit=True)
    def patched_hstack(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.compose._column_transformer', 'ColumnTransformer')"""
        # pylint: disable=no-method-argument, unused-argument, too-many-locals
        original = gorilla.get_original_attribute(
            compose.ColumnTransformer, "_hstack"
        )

        if not call_info_singleton.column_transformer_active:
            return original(self, *args, **kwargs)

        input_tuple = args[0]
        function_info = FunctionInfo(
            "sklearn.compose._column_transformer", "ColumnTransformer"
        )
        input_infos = []
        for input_df_obj in input_tuple:
            input_info = get_input_info(
                input_df_obj,
                self.mlinspect_filename,
                self.mlinspect_lineno,
                function_info,
                self.mlinspect_optional_code_reference,
                self.mlinspect_optional_source_code,
            )
            input_infos.append(input_info)

        operator_context = OperatorContext(
            OperatorType.CONCATENATION, function_info
        )
        input_annotated_dfs = [
            input_info.annotated_dfobject for input_info in input_infos
        ]
        backend_input_infos = SklearnBackend().before_call(
            operator_context, input_annotated_dfs
        )
        # No input_infos copy needed because it's only a selection and the rows not being removed don't change
        result = original(self, *args, **kwargs)
        backend_result = SklearnBackend().after_call(
            operator_context, backend_input_infos, result
        )
        result = backend_result.annotated_dfobject.result_data

        dag_node = DagNode(
            singleton.get_next_op_id(),
            BasicCodeLocation(self.mlinspect_filename, self.mlinspect_lineno),
            operator_context,
            DagNodeDetails(None, ["array"]),
            get_optional_code_info_or_none(
                self.mlinspect_optional_code_reference,
                self.mlinspect_optional_source_code,
            ),
        )
        input_dag_nodes = [input_info.dag_node for input_info in input_infos]
        add_dag_node(dag_node, input_dag_nodes, backend_result)

        return result


@gorilla.patches(preprocessing.StandardScaler)
class SklearnStandardScalerPatching:
    """Patches for sklearn StandardScaler"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("__init__")
    @gorilla.settings(allow_hit=True)
    def patched__init__(
        self,
        *,
        copy: bool = True,
        with_mean: bool = True,
        with_std: bool = True,
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str = "unknown",
        mlinspect_fit_transform_active: bool = False,
    ) -> Any:
        """Patch for ('sklearn.preprocessing._data', 'StandardScaler')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(
            preprocessing.StandardScaler, "__init__"
        )

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = (
            mlinspect_optional_code_reference
        )
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active

        self.mlinspect_non_data_func_args = {
            "copy": copy,
            "with_mean": with_mean,
            "with_std": with_std,
        }

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> None:
            """Execute inspections, add DAG node"""
            original(self, copy=copy, with_mean=with_mean, with_std=with_std)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(
            original,
            execute_inspections,
            self,
            **self.mlinspect_non_data_func_args,
        )

    @gorilla.name("fit_transform")
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.preprocessing._data.StandardScaler', 'fit_transform')"""
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = (
            True  # pylint: disable=attribute-defined-outside-init
        )
        original = gorilla.get_original_attribute(
            preprocessing.StandardScaler, "fit_transform"
        )
        function_info = FunctionInfo(
            "sklearn.preprocessing._data", "StandardScaler"
        )
        if self.mlinspect_optional_source_code == "unknown":
            new_return_value = original(self, *args, **kwargs)
        else:
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
                DagNodeDetails("Standard Scaler: fit_transform", ["array"]),
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

    @gorilla.name("transform")
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.preprocessing._data.StandardScaler', 'transform')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            preprocessing.StandardScaler, "transform"
        )
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo(
                "sklearn.preprocessing._data", "StandardScaler"
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
                DagNodeDetails("Standard Scaler: transform", ["array"]),
                get_optional_code_info_or_none(
                    self.mlinspect_optional_code_reference,
                    self.mlinspect_optional_source_code,
                ),
            )
            add_dag_node(dag_node, [input_info.dag_node], backend_result)
        else:
            new_return_value = original(self, *args, **kwargs)
        return new_return_value


@gorilla.patches(text.HashingVectorizer)
class SklearnHasingVectorizerPatching:
    """Patches for sklearn StandardScaler"""

    # pylint: disable=too-few-public-methods, redefined-builtin, too-many-locals

    @gorilla.name("__init__")
    @gorilla.settings(allow_hit=True)
    def patched__init__(
        self,
        *,
        input: str = "content",
        encoding: str = "utf-8",
        decode_error: str = "strict",
        strip_accents: str | Callable | None = None,
        lowercase: bool = True,
        preprocessor: Callable | None = None,
        tokenizer: Callable | None = None,
        stop_words: list | str | None = None,
        token_pattern: str | None = r"(?u)\b\w\w+\b",
        ngram_range: tuple = (1, 1),
        analyzer: str | Callable = "word",
        n_features: int = (2**20),
        binary: bool = False,
        norm: str = "l2",
        alternate_sign: bool = True,
        dtype: Any = numpy.float64,
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str | None = None,
        mlinspect_fit_transform_active: bool = False,
    ) -> Any:
        """Patch for ('sklearn.feature_extraction.text', 'HashingVectorizer')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(
            text.HashingVectorizer, "__init__"
        )

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = (
            mlinspect_optional_code_reference
        )
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active

        self.mlinspect_non_data_func_args = {
            "input": input,
            "encoding": encoding,
            "decode_error": decode_error,
            "strip_accents": strip_accents,
            "lowercase": lowercase,
            "preprocessor": preprocessor,
            "tokenizer": tokenizer,
            "stop_words": stop_words,
            "token_pattern": token_pattern,
            "ngram_range": ngram_range,
            "analyzer": analyzer,
            "n_features": n_features,
            "binary": binary,
            "norm": norm,
            "alternate_sign": alternate_sign,
            "dtype": dtype,
        }

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> None:
            """Execute inspections, add DAG node"""
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(
            original,
            execute_inspections,
            self,
            **self.mlinspect_non_data_func_args,
        )

    @gorilla.name("fit_transform")
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.feature_extraction.text.HashingVectorizer', 'fit_transform')"""
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = (
            True  # pylint: disable=attribute-defined-outside-init
        )
        original = gorilla.get_original_attribute(
            text.HashingVectorizer, "fit_transform"
        )
        function_info = FunctionInfo(
            "sklearn.feature_extraction.text", "HashingVectorizer"
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
        dag_node = DagNode(
            singleton.get_next_op_id(),
            BasicCodeLocation(
                self.mlinspect_caller_filename, self.mlinspect_lineno
            ),
            operator_context,
            DagNodeDetails("Hashing Vectorizer: fit_transform", ["array"]),
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

    @gorilla.name("transform")
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.preprocessing._data.StandardScaler', 'transform')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            text.HashingVectorizer, "transform"
        )
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo(
                "sklearn.feature_extraction.text", "HashingVectorizer"
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
            dag_node = DagNode(
                singleton.get_next_op_id(),
                BasicCodeLocation(
                    self.mlinspect_caller_filename, self.mlinspect_lineno
                ),
                operator_context,
                DagNodeDetails("Hashing Vectorizer: transform", ["array"]),
                get_optional_code_info_or_none(
                    self.mlinspect_optional_code_reference,
                    self.mlinspect_optional_source_code,
                ),
            )
            add_dag_node(dag_node, [input_info.dag_node], backend_result)
        else:
            new_return_value = original(self, *args, **kwargs)
        return new_return_value


@gorilla.patches(preprocessing.KBinsDiscretizer)
class SklearnKBinsDiscretizerPatching:
    """Patches for sklearn KBinsDiscretizer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("__init__")
    @gorilla.settings(allow_hit=True)
    def patched__init__(
        self,
        n_bins: int = 5,
        *,
        encode: str = "onehot",
        strategy: str = "quantile",
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str | None = None,
        mlinspect_fit_transform_active: bool = False,
    ) -> Any:
        """Patch for ('sklearn.preprocessing._discretization', 'KBinsDiscretizer')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(
            preprocessing.KBinsDiscretizer, "__init__"
        )

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = (
            mlinspect_optional_code_reference
        )
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active

        self.mlinspect_non_data_func_args = {
            "n_bins": n_bins,
            "encode": encode,
            "strategy": strategy,
        }

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> None:
            """Execute inspections, add DAG node"""
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(
            original,
            execute_inspections,
            self,
            **self.mlinspect_non_data_func_args,
        )

    @gorilla.name("fit_transform")
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.preprocessing._discretization.KBinsDiscretizer', 'fit_transform')"""
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = (
            True  # pylint: disable=attribute-defined-outside-init
        )
        original = gorilla.get_original_attribute(
            preprocessing.KBinsDiscretizer, "fit_transform"
        )
        function_info = FunctionInfo(
            "sklearn.preprocessing._discretization", "KBinsDiscretizer"
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
            DagNodeDetails("K-Bins Discretizer: fit_transform", ["array"]),
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

    @gorilla.name("transform")
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.preprocessing._discretization.KBinsDiscretizer', 'transform')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            preprocessing.KBinsDiscretizer, "transform"
        )
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo(
                "sklearn.preprocessing._discretization", "KBinsDiscretizer"
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
                DagNodeDetails("K-Bins Discretizer: transform", ["array"]),
                get_optional_code_info_or_none(
                    self.mlinspect_optional_code_reference,
                    self.mlinspect_optional_source_code,
                ),
            )
            add_dag_node(dag_node, [input_info.dag_node], backend_result)
        else:
            new_return_value = original(self, *args, **kwargs)
        return new_return_value


@gorilla.patches(preprocessing.OneHotEncoder)
class SklearnOneHotEncoderPatching:
    """Patches for sklearn OneHotEncoder"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("__init__")
    @gorilla.settings(allow_hit=True)
    def patched__init__(
        self,
        *,
        categories: str | list = "auto",
        drop: Any | None = None,
        sparse: bool = True,
        dtype: Any = numpy.float64,
        handle_unknown: str | dict = "error",
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str | None = None,
        mlinspect_fit_transform_active: bool = False,
    ) -> Any:
        """Patch for ('sklearn.preprocessing._encoders', 'OneHotEncoder')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(
            preprocessing.OneHotEncoder, "__init__"
        )

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = (
            mlinspect_optional_code_reference
        )
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active

        self.mlinspect_non_data_func_args = {
            "categories": categories,
            "drop": drop,
            "sparse": sparse,
            "dtype": dtype,
            "handle_unknown": handle_unknown,
        }

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> None:
            """Execute inspections, add DAG node"""
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(
            original,
            execute_inspections,
            self,
            **self.mlinspect_non_data_func_args,
        )

    @gorilla.name("fit_transform")
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.preprocessing._encoders.OneHotEncoder', 'fit_transform')"""
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = (
            True  # pylint: disable=attribute-defined-outside-init
        )
        original = gorilla.get_original_attribute(
            preprocessing.OneHotEncoder, "fit_transform"
        )
        function_info = FunctionInfo(
            "sklearn.preprocessing._encoders", "OneHotEncoder"
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
        dag_node = DagNode(
            singleton.get_next_op_id(),
            BasicCodeLocation(
                self.mlinspect_caller_filename, self.mlinspect_lineno
            ),
            operator_context,
            DagNodeDetails("One-Hot Encoder: fit_transform", ["array"]),
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

    @gorilla.name("transform")
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.preprocessing._encoders.OneHotEncoder', 'transform')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            preprocessing.OneHotEncoder, "transform"
        )
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo(
                "sklearn.preprocessing._encoders", "OneHotEncoder"
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
            dag_node = DagNode(
                singleton.get_next_op_id(),
                BasicCodeLocation(
                    self.mlinspect_caller_filename, self.mlinspect_lineno
                ),
                operator_context,
                DagNodeDetails("One-Hot Encoder: transform", ["array"]),
                get_optional_code_info_or_none(
                    self.mlinspect_optional_code_reference,
                    self.mlinspect_optional_source_code,
                ),
            )
            add_dag_node(dag_node, [input_info.dag_node], backend_result)
        else:
            new_return_value = original(self, *args, **kwargs)
        return new_return_value


@gorilla.patches(impute.SimpleImputer)
class SklearnSimpleImputerPatching:
    """Patches for sklearn SimpleImputer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("__init__")
    @gorilla.settings(allow_hit=True)
    def patched__init__(
        self,
        *,
        missing_values: Any = numpy.nan,
        strategy: str = "mean",
        fill_value: str | int | None = None,
        copy: bool = True,
        add_indicator: bool = False,
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str | None = None,
        mlinspect_fit_transform_active: bool = False,
    ) -> Any:
        """Patch for ('sklearn.impute._base', 'SimpleImputer')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(
            impute.SimpleImputer, "__init__"
        )

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = (
            mlinspect_optional_code_reference
        )
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active

        self.mlinspect_non_data_func_args = {
            "missing_values": missing_values,
            "strategy": strategy,
            "fill_value": fill_value,
            "copy": copy,
            "add_indicator": add_indicator,
        }

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> None:
            """Execute inspections, add DAG node"""
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(
            original,
            execute_inspections,
            self,
            **self.mlinspect_non_data_func_args,
        )

    @gorilla.name("fit_transform")
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.impute._base.SimpleImputer', 'fit_transform')"""
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = (
            True  # pylint: disable=attribute-defined-outside-init
        )
        original = gorilla.get_original_attribute(
            impute.SimpleImputer, "fit_transform"
        )
        function_info = FunctionInfo("sklearn.impute._base", "SimpleImputer")
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
        if isinstance(input_infos[0].result_data, pandas.DataFrame):
            columns = list(input_infos[0].result_data.columns)
        else:
            columns = ["array"]

        dag_node = DagNode(
            singleton.get_next_op_id(),
            BasicCodeLocation(
                self.mlinspect_caller_filename, self.mlinspect_lineno
            ),
            operator_context,
            DagNodeDetails("Simple Imputer: fit_transform", columns),
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

    @gorilla.name("transform")
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.impute._base.SimpleImputer', 'transform')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            impute.SimpleImputer, "transform"
        )
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo(
                "sklearn.impute._base", "SimpleImputer"
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
            if isinstance(input_infos[0].result_data, pandas.DataFrame):
                columns = list(input_infos[0].result_data.columns)
            else:
                columns = ["array"]

            dag_node = DagNode(
                singleton.get_next_op_id(),
                BasicCodeLocation(
                    self.mlinspect_caller_filename, self.mlinspect_lineno
                ),
                operator_context,
                DagNodeDetails("Simple Imputer: transform", columns),
                get_optional_code_info_or_none(
                    self.mlinspect_optional_code_reference,
                    self.mlinspect_optional_source_code,
                ),
            )
            add_dag_node(dag_node, [input_info.dag_node], backend_result)
        else:
            new_return_value = original(self, *args, **kwargs)
        return new_return_value


@gorilla.patches(preprocessing.FunctionTransformer)
class SklearnFunctionTransformerPatching:
    """Patches for sklearn FunctionTransformer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("__init__")
    @gorilla.settings(allow_hit=True)
    def patched__init__(
        self,
        func: Union[Callable, None] = None,
        inverse_func: Union[Callable, None] = None,
        *,
        validate: bool = False,
        accept_sparse: bool = False,
        check_inverse: bool = True,
        kw_args: dict | None = None,
        inv_kw_args: dict | None = None,
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str | None = None,
        mlinspect_fit_transform_active: bool = True,
    ) -> Any:
        """Patch for ('sklearn.preprocessing_function_transformer', 'FunctionTransformer')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(
            preprocessing.FunctionTransformer, "__init__"
        )

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = (
            mlinspect_optional_code_reference
        )
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active

        self.mlinspect_non_data_func_args = {
            "validate": validate,
            "accept_sparse": accept_sparse,
            "check_inverse": check_inverse,
            "kw_args": kw_args,
            "inv_kw_args": inv_kw_args,
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
                self,
                func=func,
                inverse_func=inverse_func,
                **self.mlinspect_non_data_func_args,
            )

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(
            original,
            execute_inspections,
            self,
            func=func,
            inverse_func=inverse_func,
            **self.mlinspect_non_data_func_args,
        )

    @gorilla.name("fit_transform")
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.preprocessing_function_transformer.FunctionTransformer', 'fit_transform')"""
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = (
            True  # pylint: disable=attribute-defined-outside-init
        )
        original = gorilla.get_original_attribute(
            preprocessing.FunctionTransformer, "fit_transform"
        )
        function_info = FunctionInfo(
            "sklearn.preprocessing_function_transformer", "FunctionTransformer"
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
        if isinstance(input_infos[0].result_data, pandas.DataFrame):
            columns = list(input_infos[0].result_data.columns)
        else:
            columns = ["array"]

        dag_node = DagNode(
            singleton.get_next_op_id(),
            BasicCodeLocation(
                self.mlinspect_caller_filename, self.mlinspect_lineno
            ),
            operator_context,
            DagNodeDetails("Function Transformer: fit_transform", columns),
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

    @gorilla.name("transform")
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.preprocessing_function_transformer.FunctionTransformer', 'transform')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            preprocessing.FunctionTransformer, "transform"
        )
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo(
                "sklearn.preprocessing_function_transformer",
                "FunctionTransformer",
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
            if isinstance(input_infos[0].result_data, pandas.DataFrame):
                columns = list(input_infos[0].result_data.columns)
            else:
                columns = ["array"]

            dag_node = DagNode(
                singleton.get_next_op_id(),
                BasicCodeLocation(
                    self.mlinspect_caller_filename, self.mlinspect_lineno
                ),
                operator_context,
                DagNodeDetails("Function Transformer: transform", columns),
                get_optional_code_info_or_none(
                    self.mlinspect_optional_code_reference,
                    self.mlinspect_optional_source_code,
                ),
            )
            add_dag_node(dag_node, [input_info.dag_node], backend_result)
        else:
            new_return_value = original(self, *args, **kwargs)
        return new_return_value


@gorilla.patches(tree.DecisionTreeClassifier)
class SklearnDecisionTreePatching:
    """Patches for sklearn DecisionTree"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("__init__")
    @gorilla.settings(allow_hit=True)
    def patched__init__(
        self,
        *,
        criterion: str = "gini",
        splitter: str = "best",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: int | float | str | None = None,
        random_state: int | Any | None = None,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        class_weight: dict | List[dict] | str | None = None,
        ccp_alpha: float = 0.0,
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str | None = None,
        mlinspect_estimator_node_id: int | None = None,
    ) -> Any:
        """Patch for ('sklearn.tree._classes', 'DecisionTreeClassifier')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals
        original = gorilla.get_original_attribute(
            tree.DecisionTreeClassifier, "__init__"
        )

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = (
            mlinspect_optional_code_reference
        )
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_estimator_node_id = mlinspect_estimator_node_id

        self.mlinspect_non_data_func_args = {
            "criterion": criterion,
            "splitter": splitter,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "max_features": max_features,
            "random_state": random_state,
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease,
            "class_weight": class_weight,
            "ccp_alpha": ccp_alpha,
        }

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> None:
            """Execute inspections, add DAG node"""
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code
            self.mlinspect_estimator_node_id = -1

        return execute_patched_func_no_op_id(
            original,
            execute_inspections,
            self,
            **self.mlinspect_non_data_func_args,
        )

    @gorilla.name("fit")
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.tree._classes.DecisionTreeClassifier', 'fit')"""
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(
            tree.DecisionTreeClassifier, "fit"
        )
        if not call_info_singleton.param_search_active:
            function_info = FunctionInfo(
                "sklearn.tree._classes", "DecisionTreeClassifier"
            )
            data_backend_result, train_data_node, train_data_result = (
                add_train_data_node(self, args[0], function_info)
            )
            (
                label_backend_result,
                train_labels_node,
                train_labels_result,
            ) = add_train_label_node(self, args[1], function_info)

            # Estimator
            operator_context = OperatorContext(
                OperatorType.ESTIMATOR, function_info
            )
            input_dfs = [
                data_backend_result.annotated_dfobject,
                label_backend_result.annotated_dfobject,
            ]
            input_infos = SklearnBackend().before_call(
                operator_context, input_dfs
            )
            result = original(
                self,
                train_data_result,
                train_labels_result,
                *args[2:],
                **kwargs,
            )
            estimator_backend_result = SklearnBackend().after_call(
                operator_context,
                input_infos,
                result,
                self.mlinspect_non_data_func_args,
            )

            self.mlinspect_estimator_node_id = (
                singleton.get_next_op_id()
            )  # pylint: disable=attribute-defined-outside-init
            dag_node = DagNode(
                self.mlinspect_estimator_node_id,
                BasicCodeLocation(
                    self.mlinspect_caller_filename, self.mlinspect_lineno
                ),
                operator_context,
                DagNodeDetails("Decision Tree", []),
                get_optional_code_info_or_none(
                    self.mlinspect_optional_code_reference,
                    self.mlinspect_optional_source_code,
                ),
            )
            add_dag_node(
                dag_node,
                [train_data_node, train_labels_node],
                estimator_backend_result,
            )
        else:
            original(self, *args, **kwargs)
        return self

    @gorilla.name("score")
    @gorilla.settings(allow_hit=True)
    def patched_score(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.tree._classes.DecisionTreeClassifier', 'score')"""

        # pylint: disable=no-method-argument
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
                "sklearn.tree._classes.DecisionTreeClassifier", "score"
            )
            data_backend_result, test_data_node, test_data_result = (
                add_test_data_dag_node(
                    args[0],
                    function_info,
                    lineno,
                    optional_code_reference,
                    optional_source_code,
                    caller_filename,
                )
            )
            label_backend_result, test_labels_node, test_labels_result = (
                add_test_label_node(
                    args[1],
                    caller_filename,
                    function_info,
                    lineno,
                    optional_code_reference,
                    optional_source_code,
                )
            )

            operator_context = OperatorContext(
                OperatorType.SCORE, function_info
            )
            input_dfs = [
                data_backend_result.annotated_dfobject,
                label_backend_result.annotated_dfobject,
            ]
            input_infos = SklearnBackend().before_call(
                operator_context, input_dfs
            )

            # Same as original, but captures the test set predictions
            predictions = self.predict(  # type: ignore[attr-defined]
                test_data_result
            )  # pylint: disable=no-member
            result = accuracy_score(test_labels_result, predictions, **kwargs)

            estimator_backend_result = SklearnBackend().after_call(
                operator_context,
                input_infos,
                predictions,
                self.mlinspect_non_data_func_args,
            )

            dag_node = DagNode(
                singleton.get_next_op_id(),
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails("Decision Tree", []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            estimator_dag_node = get_dag_node_for_id(
                self.mlinspect_estimator_node_id
            )
            add_dag_node(
                dag_node,
                [estimator_dag_node, test_data_node, test_labels_node],
                estimator_backend_result,
            )
            return result

        if not call_info_singleton.param_search_active:
            new_result = execute_patched_func_indirect_allowed(
                execute_inspections
            )
        else:
            original = gorilla.get_original_attribute(
                tree.DecisionTreeClassifier, "score"
            )
            new_result = original(self, *args, **kwargs)
        return new_result


@gorilla.patches(linear_model.SGDClassifier)
class SklearnSGDClassifierPatching:
    """Patches for sklearn SGDClassifier"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("__init__")
    @gorilla.settings(allow_hit=True)
    def patched__init__(
        self,
        loss: str = "hinge",
        *,
        penalty: str = "l2",
        alpha: float = 0.0001,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-3,
        shuffle: bool = True,
        verbose: int = 0,
        epsilon: float = DEFAULT_EPSILON,
        n_jobs: int | None = None,
        random_state: int | Any | None = None,
        learning_rate: str = "optimal",
        eta0: float = 0.0,
        power_t: float = 0.5,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 5,
        class_weight: dict | str | None = None,
        warm_start: bool = False,
        average: bool = False,
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str | None = None,
        mlinspect_estimator_node_id: int = -1,
    ) -> Any:
        """Patch for ('sklearn.linear_model._stochastic_gradient', 'SGDClassifier')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals
        original = gorilla.get_original_attribute(
            linear_model.SGDClassifier, "__init__"
        )

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = (
            mlinspect_optional_code_reference
        )
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_estimator_node_id = mlinspect_estimator_node_id

        self.mlinspect_non_data_func_args = {
            "loss": loss,
            "penalty": penalty,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "fit_intercept": fit_intercept,
            "max_iter": max_iter,
            "tol": tol,
            "shuffle": shuffle,
            "verbose": verbose,
            "epsilon": epsilon,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "learning_rate": learning_rate,
            "eta0": eta0,
            "power_t": power_t,
            "early_stopping": early_stopping,
            "validation_fraction": validation_fraction,
            "n_iter_no_change": n_iter_no_change,
            "class_weight": class_weight,
            "warm_start": warm_start,
            "average": average,
        }

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> None:
            """Execute inspections, add DAG node"""
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code
            self.mlinspect_estimator_node_id = -1

        return execute_patched_func_no_op_id(
            original,
            execute_inspections,
            self,
            **self.mlinspect_non_data_func_args,
        )

    @gorilla.name("fit")
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.linear_model._stochastic_gradient', 'fit')"""
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(
            linear_model.SGDClassifier, "fit"
        )
        if not call_info_singleton.param_search_active:
            function_info = FunctionInfo(
                "sklearn.linear_model._stochastic_gradient", "SGDClassifier"
            )
            data_backend_result, train_data_node, train_data_result = (
                add_train_data_node(self, args[0], function_info)
            )
            (
                label_backend_result,
                train_labels_node,
                train_labels_result,
            ) = add_train_label_node(self, args[1], function_info)
            # Estimator
            operator_context = OperatorContext(
                OperatorType.ESTIMATOR, function_info
            )
            input_dfs = [
                data_backend_result.annotated_dfobject,
                label_backend_result.annotated_dfobject,
            ]
            input_infos = SklearnBackend().before_call(
                operator_context, input_dfs
            )
            result = original(
                self,
                train_data_result,
                train_labels_result,
                *args[2:],
                **kwargs,
            )
            estimator_backend_result = SklearnBackend().after_call(
                operator_context,
                input_infos,
                result,
                self.mlinspect_non_data_func_args,
            )
            self.mlinspect_estimator_node_id = (
                singleton.get_next_op_id()
            )  # pylint: disable=attribute-defined-outside-init
            dag_node = DagNode(
                self.mlinspect_estimator_node_id,
                BasicCodeLocation(
                    self.mlinspect_caller_filename, self.mlinspect_lineno
                ),
                operator_context,
                DagNodeDetails("SGD Classifier", []),
                get_optional_code_info_or_none(
                    self.mlinspect_optional_code_reference,
                    self.mlinspect_optional_source_code,
                ),
            )
            add_dag_node(
                dag_node,
                [train_data_node, train_labels_node],
                estimator_backend_result,
            )
        else:
            original(self, *args, **kwargs)
        return self

    @gorilla.name("score")
    @gorilla.settings(allow_hit=True)
    def patched_score(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.linear_model._stochastic_gradient.SGDClassifier', 'score')"""

        # pylint: disable=no-method-argument
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
                "sklearn.linear_model._stochastic_gradient.SGDClassifier",
                "score",
            )
            # Test data
            data_backend_result, test_data_node, test_data_result = (
                add_test_data_dag_node(
                    args[0],
                    function_info,
                    lineno,
                    optional_code_reference,
                    optional_source_code,
                    caller_filename,
                )
            )

            # Test labels
            label_backend_result, test_labels_node, test_labels_result = (
                add_test_label_node(
                    args[1],
                    caller_filename,
                    function_info,
                    lineno,
                    optional_code_reference,
                    optional_source_code,
                )
            )

            # Score
            operator_context = OperatorContext(
                OperatorType.SCORE, function_info
            )
            input_dfs = [
                data_backend_result.annotated_dfobject,
                label_backend_result.annotated_dfobject,
            ]
            input_infos = SklearnBackend().before_call(
                operator_context, input_dfs
            )

            # Same as original, but captures the test set predictions
            predictions = self.predict(  # type: ignore[attr-defined]
                test_data_result
            )  # pylint: disable=no-member
            result = accuracy_score(test_labels_result, predictions, **kwargs)

            estimator_backend_result = SklearnBackend().after_call(
                operator_context,
                input_infos,
                predictions,
                self.mlinspect_non_data_func_args,
            )

            dag_node = DagNode(
                singleton.get_next_op_id(),
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails("SGD Classifier", []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            estimator_dag_node = get_dag_node_for_id(
                self.mlinspect_estimator_node_id
            )
            add_dag_node(
                dag_node,
                [estimator_dag_node, test_data_node, test_labels_node],
                estimator_backend_result,
            )
            return result

        if not call_info_singleton.param_search_active:
            new_result = execute_patched_func_indirect_allowed(
                execute_inspections
            )
        else:
            original = gorilla.get_original_attribute(
                linear_model.SGDClassifier, "score"
            )
            new_result = original(self, *args, **kwargs)
        return new_result


@gorilla.patches(linear_model.LogisticRegression)
class SklearnLogisticRegressionPatching:
    """Patches for sklearn LogisticRegression"""

    # pylint: disable=too-few-public-methods

    @gorilla.name("__init__")
    @gorilla.settings(allow_hit=True)
    def patched__init__(
        self,
        penalty: str = "l2",
        *,
        dual: bool = False,
        tol: float = 1e-4,
        C: float = 1.0,  # pylint: disable=invalid-name
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: dict | str | None = None,
        random_state: int | Any | None = None,
        solver: str = "lbfgs",
        max_iter: int = 100,
        multi_class: str = "auto",
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: int | None = None,
        l1_ratio: float | None = None,
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str | None = None,
        mlinspect_estimator_node_id: int = -1,
    ) -> Any:
        """Patch for ('sklearn.linear_model._logistic', 'LogisticRegression')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals
        original = gorilla.get_original_attribute(
            linear_model.LogisticRegression, "__init__"
        )

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = (
            mlinspect_optional_code_reference
        )
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_estimator_node_id = mlinspect_estimator_node_id

        self.mlinspect_non_data_func_args = {
            "penalty": penalty,
            "dual": dual,
            "tol": tol,
            "C": C,
            "fit_intercept": fit_intercept,
            "intercept_scaling": intercept_scaling,
            "class_weight": class_weight,
            "random_state": random_state,
            "solver": solver,
            "max_iter": max_iter,
            "multi_class": multi_class,
            "verbose": verbose,
            "warm_start": warm_start,
            "n_jobs": n_jobs,
            "l1_ratio": l1_ratio,
        }

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> None:
            """Execute inspections, add DAG node"""
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(
            original,
            execute_inspections,
            self,
            **self.mlinspect_non_data_func_args,
        )

    @gorilla.name("fit")
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.linear_model._logistic.LogisticRegression', 'fit')"""
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(
            linear_model.LogisticRegression, "fit"
        )
        if not call_info_singleton.param_search_active:
            function_info = FunctionInfo(
                "sklearn.linear_model._logistic", "LogisticRegression"
            )
            data_backend_result, train_data_node, train_data_result = (
                add_train_data_node(self, args[0], function_info)
            )
            (
                label_backend_result,
                train_labels_node,
                train_labels_result,
            ) = add_train_label_node(self, args[1], function_info)

            # Estimator
            operator_context = OperatorContext(
                OperatorType.ESTIMATOR, function_info
            )
            input_dfs = [
                data_backend_result.annotated_dfobject,
                label_backend_result.annotated_dfobject,
            ]
            input_infos = SklearnBackend().before_call(
                operator_context, input_dfs
            )
            result = original(
                self,
                train_data_result,
                train_labels_result,
                *args[2:],
                **kwargs,
            )
            estimator_backend_result = SklearnBackend().after_call(
                operator_context,
                input_infos,
                result,
                self.mlinspect_non_data_func_args,
            )
            self.mlinspect_estimator_node_id = (
                singleton.get_next_op_id()
            )  # pylint: disable=attribute-defined-outside-init
            dag_node = DagNode(
                self.mlinspect_estimator_node_id,
                BasicCodeLocation(
                    self.mlinspect_caller_filename, self.mlinspect_lineno
                ),
                operator_context,
                DagNodeDetails("Logistic Regression", []),
                get_optional_code_info_or_none(
                    self.mlinspect_optional_code_reference,
                    self.mlinspect_optional_source_code,
                ),
            )
            add_dag_node(
                dag_node,
                [train_data_node, train_labels_node],
                estimator_backend_result,
            )
        else:
            original(self, *args, **kwargs)
        return self

    @gorilla.name("score")
    @gorilla.settings(allow_hit=True)
    def patched_score(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('sklearn.linear_model._logistic.LogisticRegression', 'score')"""

        # pylint: disable=no-method-argument
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
                "sklearn.linear_model._logistic.LogisticRegression", "score"
            )
            # Test data
            data_backend_result, test_data_node, test_data_result = (
                add_test_data_dag_node(
                    args[0],
                    function_info,
                    lineno,
                    optional_code_reference,
                    optional_source_code,
                    caller_filename,
                )
            )

            # Test labels
            label_backend_result, test_labels_node, test_labels_result = (
                add_test_label_node(
                    args[1],
                    caller_filename,
                    function_info,
                    lineno,
                    optional_code_reference,
                    optional_source_code,
                )
            )

            # Score
            operator_context = OperatorContext(
                OperatorType.SCORE, function_info
            )
            input_dfs = [
                data_backend_result.annotated_dfobject,
                label_backend_result.annotated_dfobject,
            ]
            input_infos = SklearnBackend().before_call(
                operator_context, input_dfs
            )

            # Same as original, but captures the test set predictions
            predictions = self.predict(  # type: ignore[attr-defined]
                test_data_result
            )  # pylint: disable=no-member
            result = accuracy_score(test_labels_result, predictions, **kwargs)

            estimator_backend_result = SklearnBackend().after_call(
                operator_context,
                input_infos,
                predictions,
                self.mlinspect_non_data_func_args,
            )

            dag_node = DagNode(
                singleton.get_next_op_id(),
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails("Logistic Regression", []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            estimator_dag_node = get_dag_node_for_id(
                self.mlinspect_estimator_node_id
            )
            add_dag_node(
                dag_node,
                [estimator_dag_node, test_data_node, test_labels_node],
                estimator_backend_result,
            )
            return result

        if not call_info_singleton.param_search_active:
            new_result = execute_patched_func_indirect_allowed(
                execute_inspections
            )
        else:
            original = gorilla.get_original_attribute(
                linear_model.LogisticRegression, "score"
            )
            new_result = original(self, *args, **kwargs)
        return new_result


class SklearnKerasClassifierPatching:
    """Patches for tensorflow KerasClassifier"""

    # pylint: disable=too-few-public-methods
    @gorilla.patch(
        keras_sklearn_internal.KerasClassifier,
        name="__init__",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched__init__(
        self,
        model: Union[
            None, Callable[..., tf.keras.Model], tf.keras.Model
        ] = None,
        mlinspect_caller_filename: str | None = None,
        mlinspect_lineno: int | None = None,
        mlinspect_optional_code_reference: CodeReference | None = None,
        mlinspect_optional_source_code: str | None = None,
        mlinspect_estimator_node_id: int = -1,
        **kwargs: Any,
    ) -> Any:
        """Patch for ('scikeras.wrappers', 'KerasClassifier')"""
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals, too-many-arguments
        original = gorilla.get_original_attribute(
            keras_sklearn_internal.KerasClassifier, "__init__"
        )

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = (
            mlinspect_optional_code_reference
        )
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_estimator_node_id = mlinspect_estimator_node_id

        self.mlinspect_non_data_func_args = kwargs

        def execute_inspections(
            _: Any,
            caller_filename: str,
            lineno: int,
            optional_code_reference: CodeReference,
            optional_source_code: str,
        ) -> None:
            """Execute inspections, add DAG node"""
            original(self, model=model, **kwargs)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(
            original, execute_inspections, self, model=model, **kwargs
        )

    @gorilla.patch(
        keras_sklearn_external.KerasClassifier,
        name="fit",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched_fit(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('scikeras.wrappers.KerasClassifier', 'fit')"""
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(
            keras_sklearn_external.KerasClassifier, "fit"
        )
        if not call_info_singleton.param_search_active:
            function_info = FunctionInfo(
                "scikeras.wrappers.KerasClassifier", "fit"
            )
            data_backend_result, train_data_dag_node, train_data_result = (
                add_train_data_node(self, args[0], function_info)
            )
            (
                label_backend_result,
                train_labels_dag_node,
                train_labels_result,
            ) = add_train_label_node(self, args[1], function_info)

            # Estimator
            operator_context = OperatorContext(
                OperatorType.ESTIMATOR, function_info
            )
            input_dfs = [
                data_backend_result.annotated_dfobject,
                label_backend_result.annotated_dfobject,
            ]
            input_infos = SklearnBackend().before_call(
                operator_context, input_dfs
            )
            result = original(
                self,
                train_data_result,
                train_labels_result,
                *args[2:],
                **kwargs,
            )
            estimator_backend_result = SklearnBackend().after_call(
                operator_context,
                input_infos,
                result,
                self.mlinspect_non_data_func_args,
            )
            self.mlinspect_estimator_node_id = (
                singleton.get_next_op_id()
            )  # pylint: disable=attribute-defined-outside-init
            dag_node = DagNode(
                self.mlinspect_estimator_node_id,
                BasicCodeLocation(
                    self.mlinspect_caller_filename, self.mlinspect_lineno
                ),
                operator_context,
                DagNodeDetails("Neural Network", []),
                get_optional_code_info_or_none(
                    self.mlinspect_optional_code_reference,
                    self.mlinspect_optional_source_code,
                ),
            )
            add_dag_node(
                dag_node,
                [train_data_dag_node, train_labels_dag_node],
                estimator_backend_result,
            )
            if call_info_singleton_sklearn_inspection:
                call_info_singleton_sklearn_inspection.parent_nodes = [
                    dag_node
                ]
            if call_info_singleton_alibi:
                call_info_singleton_alibi.parent_nodes_ig = [dag_node]
                call_info_singleton_alibi.parent_nodes_ale = [dag_node]
            if call_info_singleton_dale:
                call_info_singleton_dale.parent_nodes = [dag_node]
            if call_info_singleton_dalex:
                call_info_singleton_dalex.parent_nodes = [dag_node]
        else:
            original(self, *args, **kwargs)
        return self

    @gorilla.patch(
        keras_sklearn_external.KerasClassifier,
        name="score",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched_score(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('scikeras.wrappers.KerasClassifier', 'score')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            keras_sklearn_external.KerasClassifier, "score"
        )

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
                "scikeras.wrappers.KerasClassifier", "score"
            )
            # Test data
            data_backend_result, test_data_node, test_data_result = (
                add_test_data_dag_node(
                    args[0],
                    function_info,
                    lineno,
                    optional_code_reference,
                    optional_source_code,
                    caller_filename,
                )
            )

            # Test labels
            label_backend_result, test_labels_node, test_labels_result = (
                add_test_label_node(
                    args[1],
                    caller_filename,
                    function_info,
                    lineno,
                    optional_code_reference,
                    optional_source_code,
                )
            )

            # Score
            operator_context = OperatorContext(
                OperatorType.SCORE, function_info
            )
            input_dfs = [
                data_backend_result.annotated_dfobject,
                label_backend_result.annotated_dfobject,
            ]
            input_infos = SklearnBackend().before_call(
                operator_context, input_dfs
            )

            # This currently calls predict twice, but patching here is complex. Maybe revisit this in future work
            result = original(
                self, test_data_result, test_labels_result, *args[2:], **kwargs
            )

            estimator_backend_result = SklearnBackend().after_call(
                operator_context,
                input_infos,
                result,
                self.mlinspect_non_data_func_args,
            )

            dag_node = DagNode(
                singleton.get_next_op_id(),
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails("Neural Network", []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            estimator_dag_node = get_dag_node_for_id(
                self.mlinspect_estimator_node_id
            )
            add_dag_node(
                dag_node,
                [estimator_dag_node, test_data_node, test_labels_node],
                estimator_backend_result,
            )
            return result

        if not call_info_singleton.param_search_active:
            new_result = execute_patched_func_indirect_allowed(
                execute_inspections
            )
        else:
            original = gorilla.get_original_attribute(
                keras_sklearn_external.KerasClassifier, "score"
            )
            new_result = original(self, *args, **kwargs)
        return new_result

    @gorilla.patch(
        keras_sklearn_external.KerasClassifier,
        name="predict",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched_predict(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('scikeras.wrappers.KerasClassifier', 'predict')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            keras_sklearn_external.KerasClassifier, "predict"
        )

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
                "scikeras.wrappers.KerasClassifier", "predict"
            )
            # Test data
            # TODO: make it more easily readable
            if "fit" in optional_source_code:
                if len(args) > 1:
                    return original(self, X=args[1], **kwargs)
                else:
                    return original(self, X=args[0], **kwargs)
            if (
                "score" in optional_source_code
                or "shap_values" in optional_source_code
                or (
                    "predict" in optional_source_code
                    and "Explainer" not in optional_source_code
                )
                or "PartialDependenceDisplay" in optional_source_code
                or "dalex" in optional_source_code
            ):
                return original(self, *args, **kwargs)
            if "Explainer" in optional_source_code:
                data_backend_result, test_data_node, test_data_result = (
                    add_test_data_dag_node(
                        call_info_singleton_shap.actual_explainer_input,
                        function_info,
                        lineno,
                        optional_code_reference,
                        optional_source_code,
                        caller_filename,
                    )
                )
            else:
                data_backend_result, test_data_node, test_data_result = (
                    add_test_data_dag_node(
                        args[0],
                        function_info,
                        lineno,
                        optional_code_reference,
                        optional_source_code,
                        caller_filename,
                    )
                )

            operator_context = OperatorContext(
                OperatorType.PREDICT, function_info
            )
            input_dfs = [data_backend_result.annotated_dfobject]
            input_infos = SklearnBackend().before_call(
                operator_context, input_dfs
            )

            result = original(self, test_data_result, *args[2:], **kwargs)
            estimator_backend_result = SklearnBackend().after_call(
                operator_context,
                input_infos,
                result,
                self.mlinspect_non_data_func_args,
            )
            dag_node = DagNode(
                singleton.get_next_op_id(),
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails("Neural Network", []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            estimator_dag_node = get_dag_node_for_id(
                self.mlinspect_estimator_node_id
            )
            add_dag_node(
                dag_node,
                [estimator_dag_node, test_data_node],
                estimator_backend_result,
            )
            if call_info_singleton_shap.mlinspect_explainer_node_id:
                call_info_singleton_shap.parent_nodes = [
                    dag_node,
                    test_data_node,
                ]
            return result

        if not call_info_singleton.param_search_active:
            new_result = execute_patched_func_indirect_allowed(
                execute_inspections
            )
        else:
            original = gorilla.get_original_attribute(
                keras_sklearn_external.KerasClassifier, "predict"
            )
            new_result = original(self, *args, **kwargs)
        return new_result

    @gorilla.patch(
        keras_sklearn_external.KerasClassifier,
        name="predict_proba",
        settings=gorilla.Settings(allow_hit=True),
    )
    def patched_predict_proba(self, *args: Any, **kwargs: Any) -> Any:
        """Patch for ('scikeras.wrappers.KerasClassifier', 'predict_proba')"""
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(
            keras_sklearn_external.KerasClassifier, "predict_proba"
        )

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
                "scikeras.wrappers.KerasClassifier", "predict_proba"
            )
            # Test data
            if (
                "score" in optional_source_code
                or "lime" in optional_source_code
                or "fit" in optional_source_code
                or "PartialDependenceDisplay" in optional_source_code
                or "ale" in optional_source_code
                or "dalex" in optional_source_code
            ):
                return original(self, *args, **kwargs)
            if "explain_instance" in optional_source_code:
                data_backend_result, test_data_node, _ = (
                    add_test_data_dag_node(
                        call_info_singleton_lime.actual_explainer_input,
                        function_info,
                        lineno,
                        optional_code_reference,
                        optional_source_code,
                        caller_filename,
                    )
                )
            else:
                data_backend_result, test_data_node, _ = (
                    add_test_data_dag_node(
                        args[0],
                        function_info,
                        lineno,
                        optional_code_reference,
                        optional_source_code,
                        caller_filename,
                    )
                )

            operator_context = OperatorContext(
                OperatorType.PREDICT, function_info
            )
            input_dfs = [data_backend_result.annotated_dfobject]
            input_infos = SklearnBackend().before_call(
                operator_context, input_dfs
            )

            result = original(self, *args, **kwargs)
            estimator_backend_result = SklearnBackend().after_call(
                operator_context,
                input_infos,
                result,
                self.mlinspect_non_data_func_args,
            )
            dag_node = DagNode(
                singleton.get_next_op_id(),
                BasicCodeLocation(caller_filename, lineno),
                operator_context,
                DagNodeDetails("Neural Network", []),
                get_optional_code_info_or_none(
                    optional_code_reference, optional_source_code
                ),
            )
            estimator_dag_node = get_dag_node_for_id(
                self.mlinspect_estimator_node_id
            )
            add_dag_node(
                dag_node,
                [estimator_dag_node, test_data_node],
                estimator_backend_result,
            )
            if call_info_singleton_lime.mlinspect_explainer_node_id:
                call_info_singleton_lime.parent_nodes = [dag_node]
            return result

        if not call_info_singleton.param_search_active:
            new_result = execute_patched_func_indirect_allowed(
                execute_inspections
            )
        else:
            original = gorilla.get_original_attribute(
                keras_sklearn_external.KerasClassifier, "predict_proba"
            )
            new_result = original(self, *args, **kwargs)
        return new_result
