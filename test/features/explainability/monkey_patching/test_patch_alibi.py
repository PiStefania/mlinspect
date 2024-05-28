from inspect import cleandoc

import networkx
from testfixtures import compare

from ...test_utils.utils import filter_dag_for_nodes_with_ids
from mlinspect import (
    BasicCodeLocation,
    CodeReference,
    DagNode,
    DagNodeDetails,
    FunctionInfo,
    OperatorContext,
    OperatorType,
    OptionalCodeInfo,
    PipelineInspector,
)


def test_alibi_integrated_gradients_explainer_keras_classifier() -> None:
    test_code = cleandoc(
        """
                import pandas as pd
                from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
                from scikeras.wrappers import KerasClassifier
                from tensorflow.keras.layers import Dense
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.optimizers.experimental import SGD
                import tensorflow as tf
                import numpy as np
                from alibi.explainers import IntegratedGradients
                tf.random.set_seed(seed=42)
                tf.keras.utils.set_random_seed(seed=42)

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = OneHotEncoder(sparse=False).fit_transform(df[['target']])

                def create_model(input_dim):
                    clf = Sequential()
                    clf.add(Dense(2, activation='relu', input_dim=input_dim))
                    clf.add(Dense(2, activation='relu'))
                    clf.add(Dense(2, activation='softmax'))
                    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
                    return clf

                clf = KerasClassifier(model=create_model, epochs=15, batch_size=1, verbose=0, input_dim=2, loss='categorical_crossentropy')
                clf = clf.fit(train, target)

                ig = IntegratedGradients(
                    model=clf.model_,
                    method="gausslegendre",
                    n_steps=50,
                    internal_batch_size=100,
                )
                explanation = ig.explain(X=train[:1], baselines=None, target=0)
        """
    )

    inspector_result = (
        PipelineInspector.on_pipeline_from_string(test_code)
        .set_code_reference_tracking(True)
        .execute()
    )

    filter_dag_for_nodes_with_ids(inspector_result, {7, 8, 9, 10}, 11)

    expected_dag = networkx.DiGraph()
    expected_classifier = DagNode(
        7,
        BasicCodeLocation("<string-source>", 26),
        OperatorContext(
            OperatorType.ESTIMATOR,
            FunctionInfo("scikeras.wrappers.KerasClassifier", "fit"),
        ),
        DagNodeDetails("Neural Network", []),
        OptionalCodeInfo(
            CodeReference(26, 6, 26, 123),
            "KerasClassifier(model=create_model, epochs=15, batch_size=1, "
            "verbose=0, input_dim=2, loss='categorical_crossentropy')",
        ),
    )
    expected_explainer_creation = DagNode(
        8,
        BasicCodeLocation("<string-source>", 29),
        OperatorContext(
            OperatorType.CREATE_EXPLAINER,
            FunctionInfo("alibi.explainers.IntegratedGradients", "__init__"),
        ),
        DagNodeDetails("Alibi Explainer", []),
        OptionalCodeInfo(
            CodeReference(29, 5, 34, 1),
            'IntegratedGradients(\n    model=clf.model_,\n    method="gausslegendre",\n    n_steps=50,\n    internal_batch_size=100,\n)',
        ),
    )
    expected_test_data_explainability = DagNode(
        9,
        BasicCodeLocation("<string-source>", 35),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("alibi.explainers.IntegratedGradients", "explain"),
        ),
        DagNodeDetails(None, columns=["array"]),
        OptionalCodeInfo(
            CodeReference(35, 14, 35, 63),
            "ig.explain(X=train[:1], baselines=None, target=0)",
        ),
    )
    expected_explainability = DagNode(
        10,
        BasicCodeLocation("<string-source>", 35),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("alibi.explainers.IntegratedGradients", "explain"),
        ),
        DagNodeDetails("Integrated Gradients", []),
        OptionalCodeInfo(
            CodeReference(35, 14, 35, 63),
            "ig.explain(X=train[:1], baselines=None, target=0)",
        ),
    )
    expected_dag.add_edge(expected_classifier, expected_explainer_creation)
    expected_dag.add_edge(
        expected_test_data_explainability, expected_explainability
    )
    expected_dag.add_edge(expected_explainer_creation, expected_explainability)

    compare(
        networkx.to_dict_of_dicts(inspector_result.dag),
        networkx.to_dict_of_dicts(expected_dag),
    )


def test_alibi_ale_explainer_keras_classifier() -> None:
    test_code = cleandoc(
        """
                import pandas as pd
                from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
                from scikeras.wrappers import KerasClassifier
                from tensorflow.keras.layers import Dense
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.optimizers.experimental import SGD
                import tensorflow as tf
                import numpy as np
                from alibi.explainers import ALE
                tf.random.set_seed(seed=42)
                tf.keras.utils.set_random_seed(seed=42)

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = OneHotEncoder(sparse=False).fit_transform(df[['target']])

                def create_model(input_dim):
                    clf = Sequential()
                    clf.add(Dense(2, activation='relu', input_dim=input_dim))
                    clf.add(Dense(2, activation='relu'))
                    clf.add(Dense(2, activation='softmax'))
                    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
                    return clf

                clf = KerasClassifier(model=create_model, epochs=15, batch_size=1, verbose=0, input_dim=2, loss='categorical_crossentropy')
                clf = clf.fit(train, target)

                ale_explainer = ALE(
                    clf.predict_proba,
                    feature_names=["A", "B"],
                    target_names=["no", "yes"],
                )
                explanation = ale_explainer.explain(train)
        """
    )

    inspector_result = (
        PipelineInspector.on_pipeline_from_string(test_code)
        .set_code_reference_tracking(True)
        .execute()
    )

    filter_dag_for_nodes_with_ids(inspector_result, {7, 8, 9, 10}, 11)

    expected_dag = networkx.DiGraph()
    expected_classifier = DagNode(
        7,
        BasicCodeLocation("<string-source>", 26),
        OperatorContext(
            OperatorType.ESTIMATOR,
            FunctionInfo("scikeras.wrappers.KerasClassifier", "fit"),
        ),
        DagNodeDetails("Neural Network", []),
        OptionalCodeInfo(
            CodeReference(26, 6, 26, 123),
            "KerasClassifier(model=create_model, epochs=15, batch_size=1, "
            "verbose=0, input_dim=2, loss='categorical_crossentropy')",
        ),
    )
    expected_explainer_creation = DagNode(
        8,
        BasicCodeLocation("<string-source>", 29),
        OperatorContext(
            OperatorType.CREATE_EXPLAINER,
            FunctionInfo("alibi.explainers.ALE", "__init__"),
        ),
        DagNodeDetails("Alibi Explainer", []),
        OptionalCodeInfo(
            CodeReference(29, 16, 33, 1),
            'ALE(\n    clf.predict_proba,\n    feature_names=["A", "B"],\n    target_names=["no", "yes"],\n)',
        ),
    )
    expected_test_data_explainability = DagNode(
        9,
        BasicCodeLocation("<string-source>", 34),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("alibi.explainers.ALE", "explain"),
        ),
        DagNodeDetails(None, columns=["array"]),
        OptionalCodeInfo(
            CodeReference(34, 14, 34, 42),
            "ale_explainer.explain(train)",
        ),
    )
    expected_explainability = DagNode(
        10,
        BasicCodeLocation("<string-source>", 34),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("alibi.explainers.ALE", "explain"),
        ),
        DagNodeDetails("ALE", []),
        OptionalCodeInfo(
            CodeReference(34, 14, 34, 42),
            "ale_explainer.explain(train)",
        ),
    )
    expected_dag.add_edge(expected_classifier, expected_explainer_creation)
    expected_dag.add_edge(
        expected_test_data_explainability, expected_explainability
    )
    expected_dag.add_edge(expected_explainer_creation, expected_explainability)

    compare(
        networkx.to_dict_of_dicts(inspector_result.dag),
        networkx.to_dict_of_dicts(expected_dag),
    )


def test_alibi_ale_explainer_sgd_classifier() -> None:
    test_code = cleandoc(
        """
                import pandas as pd
                from sklearn.preprocessing import StandardScaler, label_binarize
                from sklearn.linear_model import SGDClassifier
                import numpy as np
                from alibi.explainers import ALE

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = SGDClassifier(loss="log_loss", random_state=42)
                clf = clf.fit(train, target)

                ale_explainer = ALE(
                    clf.predict_proba,
                    feature_names=["A", "B"],
                    target_names=["no", "yes"],
                )
                explanation = ale_explainer.explain(train)
        """
    )

    inspector_result = (
        PipelineInspector.on_pipeline_from_string(test_code)
        .set_code_reference_tracking(True)
        .execute()
    )

    filter_dag_for_nodes_with_ids(inspector_result, {7, 8, 9, 10}, 11)

    expected_dag = networkx.DiGraph()
    expected_classifier = DagNode(
        7,
        BasicCodeLocation("<string-source>", 12),
        OperatorContext(
            OperatorType.ESTIMATOR,
            FunctionInfo(
                "sklearn.linear_model._stochastic_gradient", "SGDClassifier"
            ),
        ),
        DagNodeDetails("SGD Classifier", []),
        OptionalCodeInfo(
            CodeReference(12, 6, 12, 53),
            'SGDClassifier(loss="log_loss", random_state=42)',
        ),
    )
    expected_explainer_creation = DagNode(
        8,
        BasicCodeLocation("<string-source>", 15),
        OperatorContext(
            OperatorType.CREATE_EXPLAINER,
            FunctionInfo("alibi.explainers.ALE", "__init__"),
        ),
        DagNodeDetails("Alibi Explainer", []),
        OptionalCodeInfo(
            CodeReference(15, 16, 19, 1),
            'ALE(\n    clf.predict_proba,\n    feature_names=["A", "B"],\n    target_names=["no", "yes"],\n)',
        ),
    )
    expected_test_data_explainability = DagNode(
        9,
        BasicCodeLocation("<string-source>", 20),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("alibi.explainers.ALE", "explain"),
        ),
        DagNodeDetails(None, columns=["array"]),
        OptionalCodeInfo(
            CodeReference(20, 14, 20, 42),
            "ale_explainer.explain(train)",
        ),
    )
    expected_explainability = DagNode(
        10,
        BasicCodeLocation("<string-source>", 20),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("alibi.explainers.ALE", "explain"),
        ),
        DagNodeDetails("ALE", []),
        OptionalCodeInfo(
            CodeReference(20, 14, 20, 42),
            "ale_explainer.explain(train)",
        ),
    )
    expected_dag.add_edge(expected_classifier, expected_explainer_creation)
    expected_dag.add_edge(
        expected_test_data_explainability, expected_explainability
    )
    expected_dag.add_edge(expected_explainer_creation, expected_explainability)

    compare(
        networkx.to_dict_of_dicts(inspector_result.dag),
        networkx.to_dict_of_dicts(expected_dag),
    )


def test_alibi_ale_explainer_decision_tree_classifier() -> None:
    test_code = cleandoc(
        """
                import pandas as pd
                from sklearn.preprocessing import StandardScaler, label_binarize
                from sklearn.tree import DecisionTreeClassifier
                import numpy as np
                from alibi.explainers import ALE

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = DecisionTreeClassifier()
                clf = clf.fit(train, target)

                ale_explainer = ALE(
                    clf.predict_proba,
                    feature_names=["A", "B"],
                    target_names=["no", "yes"],
                )
                explanation = ale_explainer.explain(train)
        """
    )

    inspector_result = (
        PipelineInspector.on_pipeline_from_string(test_code)
        .set_code_reference_tracking(True)
        .execute()
    )

    filter_dag_for_nodes_with_ids(inspector_result, {7, 8, 9, 10}, 11)

    expected_dag = networkx.DiGraph()
    expected_classifier = DagNode(
        7,
        BasicCodeLocation("<string-source>", 12),
        OperatorContext(
            OperatorType.ESTIMATOR,
            FunctionInfo("sklearn.tree._classes", "DecisionTreeClassifier"),
        ),
        DagNodeDetails("Decision Tree", []),
        OptionalCodeInfo(
            CodeReference(12, 6, 12, 30), "DecisionTreeClassifier()"
        ),
    )
    expected_explainer_creation = DagNode(
        8,
        BasicCodeLocation("<string-source>", 15),
        OperatorContext(
            OperatorType.CREATE_EXPLAINER,
            FunctionInfo("alibi.explainers.ALE", "__init__"),
        ),
        DagNodeDetails("Alibi Explainer", []),
        OptionalCodeInfo(
            CodeReference(15, 16, 19, 1),
            'ALE(\n    clf.predict_proba,\n    feature_names=["A", "B"],\n    target_names=["no", "yes"],\n)',
        ),
    )
    expected_test_data_explainability = DagNode(
        9,
        BasicCodeLocation("<string-source>", 20),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("alibi.explainers.ALE", "explain"),
        ),
        DagNodeDetails(None, columns=["array"]),
        OptionalCodeInfo(
            CodeReference(20, 14, 20, 42),
            "ale_explainer.explain(train)",
        ),
    )
    expected_explainability = DagNode(
        10,
        BasicCodeLocation("<string-source>", 20),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("alibi.explainers.ALE", "explain"),
        ),
        DagNodeDetails("ALE", []),
        OptionalCodeInfo(
            CodeReference(20, 14, 20, 42),
            "ale_explainer.explain(train)",
        ),
    )
    expected_dag.add_edge(expected_classifier, expected_explainer_creation)
    expected_dag.add_edge(
        expected_test_data_explainability, expected_explainability
    )
    expected_dag.add_edge(expected_explainer_creation, expected_explainability)

    compare(
        networkx.to_dict_of_dicts(inspector_result.dag),
        networkx.to_dict_of_dicts(expected_dag),
    )


def test_alibi_ale_explainer_logistic_regression() -> None:
    test_code = cleandoc(
        """
                import pandas as pd
                from sklearn.preprocessing import StandardScaler, label_binarize
                from sklearn.linear_model import LogisticRegression
                import numpy as np
                from alibi.explainers import ALE

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = LogisticRegression()
                clf = clf.fit(train, target)

                ale_explainer = ALE(
                    clf.predict_proba,
                    feature_names=["A", "B"],
                    target_names=["no", "yes"],
                )
                explanation = ale_explainer.explain(train)
        """
    )

    inspector_result = (
        PipelineInspector.on_pipeline_from_string(test_code)
        .set_code_reference_tracking(True)
        .execute()
    )

    filter_dag_for_nodes_with_ids(inspector_result, {7, 8, 9, 10}, 11)

    expected_dag = networkx.DiGraph()
    expected_classifier = DagNode(
        7,
        BasicCodeLocation("<string-source>", 12),
        OperatorContext(
            OperatorType.ESTIMATOR,
            FunctionInfo(
                "sklearn.linear_model._logistic", "LogisticRegression"
            ),
        ),
        DagNodeDetails("Logistic Regression", []),
        OptionalCodeInfo(CodeReference(12, 6, 12, 26), "LogisticRegression()"),
    )
    expected_explainer_creation = DagNode(
        8,
        BasicCodeLocation("<string-source>", 15),
        OperatorContext(
            OperatorType.CREATE_EXPLAINER,
            FunctionInfo("alibi.explainers.ALE", "__init__"),
        ),
        DagNodeDetails("Alibi Explainer", []),
        OptionalCodeInfo(
            CodeReference(15, 16, 19, 1),
            'ALE(\n    clf.predict_proba,\n    feature_names=["A", "B"],\n    target_names=["no", "yes"],\n)',
        ),
    )
    expected_test_data_explainability = DagNode(
        9,
        BasicCodeLocation("<string-source>", 20),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("alibi.explainers.ALE", "explain"),
        ),
        DagNodeDetails(None, columns=["array"]),
        OptionalCodeInfo(
            CodeReference(20, 14, 20, 42),
            "ale_explainer.explain(train)",
        ),
    )
    expected_explainability = DagNode(
        10,
        BasicCodeLocation("<string-source>", 20),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("alibi.explainers.ALE", "explain"),
        ),
        DagNodeDetails("ALE", []),
        OptionalCodeInfo(
            CodeReference(20, 14, 20, 42),
            "ale_explainer.explain(train)",
        ),
    )
    expected_dag.add_edge(expected_classifier, expected_explainer_creation)
    expected_dag.add_edge(
        expected_test_data_explainability, expected_explainability
    )
    expected_dag.add_edge(expected_explainer_creation, expected_explainability)

    compare(
        networkx.to_dict_of_dicts(inspector_result.dag),
        networkx.to_dict_of_dicts(expected_dag),
    )
