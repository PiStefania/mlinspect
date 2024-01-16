"""
Explainer Inspection
"""
from typing import Iterable, List, Optional

import numpy as np
from scikeras.wrappers import KerasClassifier

from features.explainability.inspections.explainability_methods_enum import ExplainabilityMethodsEnum
from mlinspect import OperatorType
from mlinspect.inspections import Inspection


class Explainer(Inspection):

    def __init__(self, methods: List[ExplainabilityMethodsEnum], explainer_input: np.ndarray, test_input: np.ndarray, features: List[str], nsamples:int=100,):
        self.methods = methods
        self._operator_type = None
        self._results = {}
        self.test_input = test_input
        self.features = features
        self.nsamples = nsamples
        self.explainer_input = explainer_input

    def visit_operator(self, inspection_input) -> Iterable[any]:
        """
        Visit an operator
        """
        self._operator_type = inspection_input.operator_context.operator
        if self._operator_type == OperatorType.ESTIMATOR:
            model: Optional[KerasClassifier] = None
            for row in inspection_input.row_iterator:
                model = row.output[0]
            if not model:
                yield None
            if ExplainabilityMethodsEnum.SHAP in self.methods:
                import shap
                explainer = shap.KernelExplainer(model.predict, self.explainer_input)
                results = explainer.shap_values(self.test_input, nsamples=self.nsamples)
                self._results[ExplainabilityMethodsEnum.SHAP] = {"explainer": explainer, "results": results}
            if ExplainabilityMethodsEnum.LIME in self.methods:
                import lime
                import lime.lime_tabular
                explainer = lime.lime_tabular.LimeTabularExplainer(self.explainer_input,
                                                                   mode='classification',
                                                                   feature_names=self.features,
                                                                   class_names=["label"])

                result = explainer.explain_instance(self.test_input[0], model.predict_proba)
                self._results[ExplainabilityMethodsEnum.LIME] = {"explainer": explainer, "results": result}
            if ExplainabilityMethodsEnum.PDP in self.methods:
                from sklearn.inspection import PartialDependenceDisplay
                display = PartialDependenceDisplay.from_estimator(model, self.explainer_input, features=[1,2], kind="average")
                self._results[ExplainabilityMethodsEnum.PDP] = {"explainer": None, "results": display}
            if ExplainabilityMethodsEnum.ICE in self.methods:
                from sklearn.inspection import PartialDependenceDisplay
                display = PartialDependenceDisplay.from_estimator(model, self.explainer_input, features=[1, 2],
                                                                  kind="individual")
                self._results[ExplainabilityMethodsEnum.ICE] = {"explainer": None, "results": display}
            if ExplainabilityMethodsEnum.INTEGRATED_GRADIENTS in self.methods:
                from alibi.explainers import IntegratedGradients
                # ig = IntegratedGradients(model,
                #                          layer=None,
                #                          taget_fn=None,
                #                          method="gausslegendre",
                #                          n_steps=50,
                #                          internal_batch_size=100)
                # explanation = ig.explain(X,
                #                          baselines=None,
                #                          target=None)
                #
                # attributions = explanation.attributions
                ...
            if ExplainabilityMethodsEnum.ALE in self.methods:
                from PyALE import ale
                # random.seed(123)
                # X_sample = X[features].loc[random.sample(X.index.to_list(), 1000), :]
                # ale_eff = ale(
                #     X=X_sample, model=model, feature=["carat"], grid_size=50, include_CI=True, C=0.95
                # )
                ...
            if ExplainabilityMethodsEnum.DALE in self.methods:
                from ..dale.dale import DALE
                # dale = DALE(data=X, model=model, model_jac=model_jac)
                # dale.fit(features=0, params={"method": "fixed", "nof_bins": 10})
                # dale.plot(s=0, error=False)
                ...
            if ExplainabilityMethodsEnum.DALEX in self.methods:
                import dalex as dx
                # exp = dx.Explainer(clf, X, y)
                # explanation = exp.model_parts()
                # explanation.result
                # explanation.plot()
                # exp.predict_parts(new_observation).result
                # exp.predict_parts(new_observation).plot()
                ...
        yield None

    def get_operator_annotation_after_visit(self) -> any:
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
    def inspection_id(self):
        return tuple([obj.name for obj in self.methods])

