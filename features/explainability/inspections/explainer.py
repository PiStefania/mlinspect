"""
Explainer Inspection
"""

from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from features.explainability.inspections.explainability_methods_enum import (
    ExplainabilityMethodsEnum,
)
from features.explainability.inspections.utils import (
    is_neural_network,
    is_regression,
    is_supported_estimator,
)

from mlinspect import OperatorType
from mlinspect.inspections import (
    Inspection,
    InspectionInputDataSource,
    InspectionInputNAryOperator,
    InspectionInputSinkOperator,
    InspectionInputUnaryOperator,
)
from mlinspect.inspections._inspection_input import InspectionRowSinkOperator


class Explainer(Inspection):

    # TODO: tweak input values
    def __init__(
        self,
        methods: List[ExplainabilityMethodsEnum],
        test_data: Any,
        feature_names: Optional[List[str]],
        test_labels: Optional[List[str]],
    ) -> None:
        # Inspection generic attributes
        self._operator_type: OperatorType | None = None
        self._results: Dict = {}
        # Inspection specific attributes
        self.methods = methods
        self.test_data = test_data
        self.features = feature_names
        self.test_labels = test_labels

    def visit_operator(
        self,
        inspection_input: Union[
            InspectionInputDataSource,
            InspectionInputUnaryOperator,
            InspectionInputNAryOperator,
            InspectionInputSinkOperator,
        ],
    ) -> Iterable[Any]:
        """
        Visit an operator
        """
        self._operator_type = inspection_input.operator_context.operator
        if self._operator_type == OperatorType.ESTIMATOR:
            model: Any | None = None
            train_data: Any | None = None
            train_labels: Any | None = None
            for row in inspection_input.row_iterator:
                if isinstance(row, InspectionRowSinkOperator):
                    if inspection_input.output is not None:
                        if is_supported_estimator(
                            inspection_input.output.estimator
                        ):
                            estimator_info = inspection_input.output
                            print(
                                "Estimator is supported for explainer inspection."
                            )
                            model = estimator_info.estimator
                            train_data = estimator_info.train_data
                            train_labels = estimator_info.train_labels
                        else:
                            print(
                                f"Estimator {type(inspection_input.output.estimator)} is not supported for explainer "
                                f"inspection."
                            )
            if not model:
                yield None
            if model:
                if ExplainabilityMethodsEnum.SHAP in self.methods:
                    import shap

                    explainer = shap.KernelExplainer(model.predict, train_data)
                    results = explainer.shap_values(
                        self.test_data[:2],
                    )
                    self._results[ExplainabilityMethodsEnum.SHAP] = {
                        "explainer": explainer,
                        "results": results,
                    }
                if ExplainabilityMethodsEnum.LIME in self.methods:
                    import lime
                    import lime.lime_tabular

                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        train_data,
                        mode="classification",
                        feature_names=self.features,
                        class_names=self.test_labels,
                    )

                    result = explainer.explain_instance(
                        self.test_data[0], model.predict_proba
                    )
                    self._results[ExplainabilityMethodsEnum.LIME] = {
                        "explainer": explainer,
                        "results": result,
                    }
                if ExplainabilityMethodsEnum.PDP in self.methods:
                    from sklearn.inspection import PartialDependenceDisplay

                    display = PartialDependenceDisplay.from_estimator(
                        model,
                        train_data,
                        features=[1, 2],
                        kind="average",
                    )
                    self._results[ExplainabilityMethodsEnum.PDP] = {
                        "explainer": None,
                        "results": display,
                    }
                if ExplainabilityMethodsEnum.ICE in self.methods:
                    from sklearn.inspection import PartialDependenceDisplay

                    display = PartialDependenceDisplay.from_estimator(
                        model,
                        train_data,
                        features=[1, 2],
                        kind="individual",
                    )
                    self._results[ExplainabilityMethodsEnum.ICE] = {
                        "explainer": None,
                        "results": display,
                    }
                if (
                    ExplainabilityMethodsEnum.INTEGRATED_GRADIENTS
                    in self.methods
                    and is_neural_network(model)
                ):
                    from alibi.explainers import IntegratedGradients

                    ig = IntegratedGradients(
                        model.model_,
                        method="gausslegendre",
                        n_steps=50,
                        internal_batch_size=100,
                    )
                    explanation = ig.explain(
                        self.test_data, baselines=None, target=0
                    )
                    self._results[
                        ExplainabilityMethodsEnum.INTEGRATED_GRADIENTS
                    ] = {"explainer": ig, "results": explanation}
                if ExplainabilityMethodsEnum.ALE in self.methods:
                    from alibi.explainers import ALE

                    explainer = ALE(
                        model.predict_proba,
                        feature_names=self.features,
                        target_names=self.test_labels,
                    )
                    explanation = explainer.explain(train_data)
                    self._results[ExplainabilityMethodsEnum.ALE] = {
                        "explainer": explainer,
                        "results": explanation,
                    }
                if (
                    ExplainabilityMethodsEnum.DALE in self.methods
                    and is_neural_network(model)
                ):
                    import tensorflow as tf

                    from ..dale.dale import DALE

                    def model_grad(input_value: np.ndarray) -> Any:
                        x_inp = tf.cast(input_value, tf.float32)
                        with tf.GradientTape() as tape:
                            tape.watch(x_inp)
                            preds = model.model_(x_inp)
                        grads = tape.gradient(preds, x_inp)
                        return grads.numpy()

                    dale = DALE(
                        data=train_data, model=model, model_jac=model_grad
                    )
                    dale.fit()
                    explanations = dale.eval(x=self.test_data, s=0)
                    self._results[ExplainabilityMethodsEnum.DALE] = {
                        "explainer": dale,
                        "results": explanations,
                    }
                if ExplainabilityMethodsEnum.DALEX in self.methods:
                    model_type = "classification"
                    if is_regression(model):
                        model_type = "regression"
                    import dalex as dx

                    explainer = dx.Explainer(
                        model,
                        train_data,
                        train_labels,
                        predict_function=model.predict,
                        model_type=model_type,
                    )
                    explanation = explainer.model_parts()
                    train_explanation = explanation.result
                    df = pd.DataFrame([self.test_data[0]], index=["first_row"])
                    test_explanation = explainer.predict_parts(
                        df,
                        label=df.index[0],
                        type="break_down",
                    ).result
                    self._results[ExplainabilityMethodsEnum.DALEX] = {
                        "explainer": explainer,
                        "results": {
                            "train": train_explanation,
                            "test": test_explanation,
                        },
                    }
        yield None

    def get_operator_annotation_after_visit(self) -> Any:
        assert self._operator_type
        if self._operator_type is OperatorType.ESTIMATOR:
            result = self._results
            self._operator_type = None
            self._results = {}
            return result
        self._operator_type = None
        self._results = {}
        return None

    @property
    def inspection_id(self) -> Any | None:
        return tuple([obj.name for obj in self.methods])
