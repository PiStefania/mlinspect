"""
Explainer Inspection
"""

from typing import Any, Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from scikeras.wrappers import KerasClassifier

from features.explainability.inspections.explainability_methods_enum import (
    ExplainabilityMethodsEnum,
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
        explainer_input: Any,
        test_input: Any,
        features: List[str],
        test_labels: List[str],
        train_labels: List[str],
        nsamples: int = 100,
    ) -> None:
        self.methods = methods
        self._operator_type: OperatorType | None = None
        self._results: Dict = {}
        self.test_input = test_input
        self.features = features
        self.nsamples = nsamples
        self.explainer_input = explainer_input
        self.test_labels = test_labels
        self.train_labels = train_labels

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
            # TODO: add more classifiers
            model: KerasClassifier | None = None
            for row in inspection_input.row_iterator:
                if isinstance(row, InspectionRowSinkOperator):
                    model = inspection_input.self_output
            if not model:
                yield None
            if model:
                if ExplainabilityMethodsEnum.SHAP in self.methods:
                    import shap

                    explainer = shap.KernelExplainer(
                        model.predict, self.explainer_input
                    )
                    results = explainer.shap_values(
                        self.test_input[:2], nsamples=self.nsamples
                    )
                    self._results[ExplainabilityMethodsEnum.SHAP] = {
                        "explainer": explainer,
                        "results": results,
                    }
                if ExplainabilityMethodsEnum.LIME in self.methods:
                    import lime
                    import lime.lime_tabular

                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        self.explainer_input,
                        mode="classification",
                        feature_names=self.features,
                        class_names=["label"],
                    )

                    result = explainer.explain_instance(
                        self.test_input[0], model.predict_proba
                    )
                    self._results[ExplainabilityMethodsEnum.LIME] = {
                        "explainer": explainer,
                        "results": result,
                    }
                if ExplainabilityMethodsEnum.PDP in self.methods:
                    from sklearn.inspection import PartialDependenceDisplay

                    display = PartialDependenceDisplay.from_estimator(
                        model,
                        self.explainer_input,
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
                        self.explainer_input,
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
                ):
                    from alibi.explainers import IntegratedGradients

                    ig = IntegratedGradients(
                        model.model_,
                        method="gausslegendre",
                        n_steps=50,
                        internal_batch_size=100,
                    )
                    explanation = ig.explain(
                        self.test_input, baselines=None, target=0
                    )
                    self._results[
                        ExplainabilityMethodsEnum.INTEGRATED_GRADIENTS
                    ] = {"explainer": ig, "results": explanation}
                if ExplainabilityMethodsEnum.ALE in self.methods:
                    from alibi.explainers import ALE

                    explainer = ALE(
                        model.predict_proba,
                        feature_names=self.features,
                        target_names=["label"],
                    )
                    explanation = explainer.explain(self.explainer_input)
                    self._results[ExplainabilityMethodsEnum.ALE] = {
                        "explainer": explainer,
                        "results": explanation,
                    }
                if ExplainabilityMethodsEnum.DALE in self.methods:
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
                        data=self.test_input, model=model, model_jac=model_grad
                    )
                    dale.fit()
                    explanations = dale.eval(x=self.explainer_input, s=0)
                    self._results[ExplainabilityMethodsEnum.DALE] = {
                        "explainer": dale,
                        "results": explanations,
                    }
                if ExplainabilityMethodsEnum.DALEX in self.methods:
                    import dalex as dx

                    explainer = dx.Explainer(
                        model,
                        self.explainer_input,
                        self.train_labels,
                        predict_function=model.predict,
                    )
                    explanation = explainer.model_parts()
                    train_explanation = explanation.result
                    df = pd.DataFrame(
                        [self.test_input[0]], index=["first_row"]
                    )
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
