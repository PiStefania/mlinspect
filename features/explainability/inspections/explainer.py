"""
Explainer Inspection
"""
from typing import Iterable, List

import numpy as np
from scikeras.wrappers import KerasClassifier

from features.explainability.inspections.explainability_methods_enum import ExplainabilityMethodsEnum
from mlinspect import OperatorType
from mlinspect.inspections import Inspection


class Explainer(Inspection):

    def __init__(self, methods: List[ExplainabilityMethodsEnum], explainer_input: np.ndarray, test_input: np.ndarray, nsamples:int=100):
        self.methods = methods
        self._operator_type = None
        self._explainer = None
        self._results = {}
        self.test_input = test_input
        self.nsamples = nsamples
        self.explainer_input = explainer_input

    def visit_operator(self, inspection_input) -> Iterable[any]:
        """
        Visit an operator
        """
        self._operator_type = inspection_input.operator_context.operator
        if self._operator_type == OperatorType.ESTIMATOR:
            if ExplainabilityMethodsEnum.SHAP in self.methods:
                import shap
                for row in inspection_input.row_iterator:
                    model: KerasClassifier = row.output[0]
                    self._explainer = shap.KernelExplainer(model.predict, self.explainer_input)
                    self._results[ExplainabilityMethodsEnum.SHAP] = self._explainer.shap_values(self.test_input, nsamples=self.nsamples)
            elif ExplainabilityMethodsEnum.LIME in self.methods:
                import lime
                import lime.lime_tabular
                # explainer = lime.lime_tabular.LimeTabularExplainer(df_titanic[model.feature_name()].astype(int).values,
                #                                                    mode='classification',
                #                                                    training_labels=df_titanic['Survived'],
                #                                                    feature_names=model.feature_name())
                #
                # # asking for explanation for LIME model
                # i = 1
                # exp = explainer.explain_instance(df_titanic.loc[i, feat].astype(int).values, prob, num_features=5)
                ...
            elif ExplainabilityMethodsEnum.PDP in self.methods:
                from sklearn.inspection import partial_dependence
                from sklearn.inspection import PartialDependenceDisplay
                # pd_results = partial_dependence(
                # clf, X, features = 0, kind = "average", grid_resolution = 5)
                # display = PartialDependenceDisplay(
                #     [pd_results], features=features, feature_names=feature_names,target_idx = 0, deciles = deciles
                # )
                ...
            elif ExplainabilityMethodsEnum.ICE in self.methods:
                from sklearn.inspection import partial_dependence
                from sklearn.inspection import PartialDependenceDisplay
                # pd_results = partial_dependence(
                # clf, X, features = 0, kind = "individual", grid_resolution = 5)
                # display = PartialDependenceDisplay(
                #     [pd_results], features=features, feature_names=feature_names,target_idx = 0, deciles = deciles
                # )
                ...
            elif ExplainabilityMethodsEnum.INTEGRATED_GRADIENTS in self.methods:
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
            elif ExplainabilityMethodsEnum.ALE in self.methods:
                from PyALE import ale
                # random.seed(123)
                # X_sample = X[features].loc[random.sample(X.index.to_list(), 1000), :]
                # ale_eff = ale(
                #     X=X_sample, model=model, feature=["carat"], grid_size=50, include_CI=True, C=0.95
                # )
                ...
            elif ExplainabilityMethodsEnum.DALE in self.methods:
                from ..dale.dale import DALE
                # dale = DALE(data=X, model=model, model_jac=model_jac)
                # dale.fit(features=0, params={"method": "fixed", "nof_bins": 10})
                # dale.plot(s=0, error=False)
                ...
            elif ExplainabilityMethodsEnum.DALEX in self.methods:
                import dalex as dx
                # exp = dx.Explainer(clf, X, y)
                # explanation = exp.model_parts()
                # explanation.result
                # explanation.plot()
                # exp.predict_parts(new_observation).result
                # exp.predict_parts(new_observation).plot()
                ...
            else:
                raise Exception(f"No implementation provided for {self.methods}.")
        yield None

    def get_operator_annotation_after_visit(self) -> any:
        assert self._operator_type
        if self._operator_type is OperatorType.ESTIMATOR:
            result = {"explainer": self._explainer, "results": self._results}
            self._operator_type = None
            self._explainer = None
            self._results = None
            return result
        self._operator_type = None
        self._explainer = None
        self._results = None
        return None

    @property
    def inspection_id(self):
        return tuple([obj.name for obj in self.methods])

