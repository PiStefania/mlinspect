from typing import Any, Dict

import dalex
import numpy as np
from alibi.explainers import ALE, IntegratedGradients
from lime.lime_tabular import LimeTabularExplainer
from shap import KernelExplainer

from example_pipelines._pipelines import EXPLAINABILITY_HEALTHCARE_PY
from features.explainability.dale.dale import DALE
from features.explainability.inspections.explainability_methods_enum import (
    ExplainabilityMethodsEnum,
)
from features.explainability.inspections.explainer import Explainer

from mlinspect import OperatorType, PipelineInspector


def test_explainer_healthcare_all_methods() -> None:

    inspector_result = (
        PipelineInspector.on_pipeline_from_py_file(
            EXPLAINABILITY_HEALTHCARE_PY
        )
        .add_required_inspection(Explainer(**get_explainability_method_args()))
        .execute()
    )

    assert_explainer_inspection_results(inspector_result)


def get_explainability_method_args() -> Dict[str, Any]:
    import os

    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from mlinspect.monkeypatching._mlinspect_ndarray import MlinspectNdarray
    from mlinspect.utils import get_project_root

    # prepare data for explainer
    COUNTIES_OF_INTEREST = ["county2", "county3"]

    patients = pd.read_csv(
        os.path.join(
            str(get_project_root()),
            "example_pipelines",
            "healthcare",
            "patients.csv",
        ),
        na_values="?",
    )
    histories = pd.read_csv(
        os.path.join(
            str(get_project_root()),
            "example_pipelines",
            "healthcare",
            "histories.csv",
        ),
        na_values="?",
    )

    data = patients.merge(histories, on=["ssn"])
    complications = data.groupby("age_group").agg(
        mean_complications=("complications", "mean")
    )
    data = data.merge(complications, on=["age_group"])
    data["label"] = data["complications"] > 1.2 * data["mean_complications"]
    data = data[
        [
            "smoker",
            "last_name",
            "county",
            "num_children",
            "race",
            "income",
            "label",
        ]
    ]
    data = data[data["county"].isin(COUNTIES_OF_INTEREST)]
    train_data, test_data = train_test_split(data)
    y_test = test_data["label"]
    X_test = test_data.drop("label", axis=1)

    impute_and_one_hot_encode = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(sparse=False, handle_unknown="ignore")),
        ]
    )
    featurisation = ColumnTransformer(
        transformers=[
            (
                "impute_and_one_hot_encode",
                impute_and_one_hot_encode,
                ["smoker", "county", "race"],
            ),
            ("numeric", StandardScaler(), ["num_children", "income"]),
        ],
        remainder="drop",
    )

    X_t_test: MlinspectNdarray = featurisation.fit_transform(X_test, y_test)
    return {
        "methods": [
            ExplainabilityMethodsEnum.LIME,
            ExplainabilityMethodsEnum.SHAP,
            ExplainabilityMethodsEnum.PDP,
            ExplainabilityMethodsEnum.ICE,
            ExplainabilityMethodsEnum.INTEGRATED_GRADIENTS,
            ExplainabilityMethodsEnum.ALE,
            ExplainabilityMethodsEnum.DALE,
            ExplainabilityMethodsEnum.DALEX,
        ],
        "test_data": X_t_test.view(np.ndarray),
        "feature_names": featurisation.get_feature_names_out(),
        "test_labels": ["False", "True"],
    }


def assert_explainer_inspection_results(inspector_result) -> None:
    extracted_dag = inspector_result.dag
    dag_node_to_inspection_results = (
        inspector_result.dag_node_to_inspection_results
    )
    relevant_node = [
        node
        for node in extracted_dag.nodes
        if node.operator_info.operator
        in {
            OperatorType.ESTIMATOR,
        }
    ][0]
    inspection_result = dag_node_to_inspection_results[relevant_node][
        Explainer(**get_explainability_method_args())
    ]
    assert inspection_result is not None

    assert ExplainabilityMethodsEnum.SHAP in inspection_result
    assert "explainer" in inspection_result[ExplainabilityMethodsEnum.SHAP]
    assert isinstance(
        inspection_result[ExplainabilityMethodsEnum.SHAP]["explainer"],
        KernelExplainer,
    )
    assert "results" in inspection_result[ExplainabilityMethodsEnum.SHAP]
    assert (
        inspection_result[ExplainabilityMethodsEnum.SHAP]["results"]
        is not None
    )

    assert ExplainabilityMethodsEnum.LIME in inspection_result
    assert "explainer" in inspection_result[ExplainabilityMethodsEnum.LIME]
    assert isinstance(
        inspection_result[ExplainabilityMethodsEnum.LIME]["explainer"],
        LimeTabularExplainer,
    )
    assert "results" in inspection_result[ExplainabilityMethodsEnum.LIME]
    assert (
        inspection_result[ExplainabilityMethodsEnum.LIME]["results"]
        is not None
    )

    assert ExplainabilityMethodsEnum.PDP in inspection_result
    assert "explainer" in inspection_result[ExplainabilityMethodsEnum.PDP]
    assert (
        inspection_result[ExplainabilityMethodsEnum.PDP]["explainer"] is None
    )
    assert "results" in inspection_result[ExplainabilityMethodsEnum.PDP]
    assert (
        inspection_result[ExplainabilityMethodsEnum.PDP]["results"] is not None
    )

    assert ExplainabilityMethodsEnum.ICE in inspection_result
    assert "explainer" in inspection_result[ExplainabilityMethodsEnum.ICE]
    assert (
        inspection_result[ExplainabilityMethodsEnum.ICE]["explainer"] is None
    )
    assert "results" in inspection_result[ExplainabilityMethodsEnum.ICE]
    assert (
        inspection_result[ExplainabilityMethodsEnum.ICE]["results"] is not None
    )

    assert ExplainabilityMethodsEnum.INTEGRATED_GRADIENTS in inspection_result
    assert (
        "explainer"
        in inspection_result[ExplainabilityMethodsEnum.INTEGRATED_GRADIENTS]
    )
    assert isinstance(
        inspection_result[ExplainabilityMethodsEnum.INTEGRATED_GRADIENTS][
            "explainer"
        ],
        IntegratedGradients,
    )
    assert (
        "results"
        in inspection_result[ExplainabilityMethodsEnum.INTEGRATED_GRADIENTS]
    )
    assert (
        inspection_result[ExplainabilityMethodsEnum.INTEGRATED_GRADIENTS][
            "results"
        ]
        is not None
    )

    assert ExplainabilityMethodsEnum.ALE in inspection_result
    assert "explainer" in inspection_result[ExplainabilityMethodsEnum.ALE]
    assert isinstance(
        inspection_result[ExplainabilityMethodsEnum.ALE]["explainer"], ALE
    )
    assert "results" in inspection_result[ExplainabilityMethodsEnum.ALE]
    assert (
        inspection_result[ExplainabilityMethodsEnum.ALE]["results"] is not None
    )

    assert ExplainabilityMethodsEnum.DALE in inspection_result
    assert "explainer" in inspection_result[ExplainabilityMethodsEnum.DALE]
    assert isinstance(
        inspection_result[ExplainabilityMethodsEnum.DALE]["explainer"], DALE
    )
    assert "results" in inspection_result[ExplainabilityMethodsEnum.DALE]
    assert (
        inspection_result[ExplainabilityMethodsEnum.DALE]["results"]
        is not None
    )

    assert ExplainabilityMethodsEnum.DALEX in inspection_result
    assert "explainer" in inspection_result[ExplainabilityMethodsEnum.DALEX]
    assert isinstance(
        inspection_result[ExplainabilityMethodsEnum.DALEX]["explainer"],
        dalex.Explainer,
    )
    assert "results" in inspection_result[ExplainabilityMethodsEnum.DALEX]
    assert (
        inspection_result[ExplainabilityMethodsEnum.DALEX]["results"]
        is not None
    )
