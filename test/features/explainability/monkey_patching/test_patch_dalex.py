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


def test_dalex_explainer_keras_classifier() -> None:
    test_code = cleandoc(
        """
                import pandas as pd
                from sklearn.preprocessing import StandardScaler, OneHotEncoder
                from scikeras.wrappers import KerasClassifier
                from tensorflow.keras.layers import Dense
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.optimizers.experimental import SGD
                import tensorflow as tf
                import numpy as np
                import dalex
                tf.random.set_seed(seed=42)
                tf.keras.utils.set_random_seed(seed=42)

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3]})

                train = StandardScaler().fit_transform(df)
                labels = np.array([0, 0, 1, 1])

                def create_model(input_dim):
                    clf = Sequential()
                    clf.add(Dense(2, activation='relu', input_dim=input_dim))
                    clf.add(Dense(2, activation='relu'))
                    clf.add(Dense(1, activation='sigmoid'))
                    clf.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=["accuracy"])
                    return clf

                clf = KerasClassifier(model=create_model, epochs=15, batch_size=1, verbose=0, input_dim=2, loss='binary_crossentropy')
                clf = clf.fit(train, labels)

                dalex_explainer = dalex.Explainer(
                    clf, data=train, y=labels, predict_function=KerasClassifier.predict
                )
                explanation = dalex_explainer.model_parts()
                train_explanation = explanation.result
                df = pd.DataFrame([train[0]], index=["first_row"])
                test_explanation = dalex_explainer.predict_parts(df, label=df.index[0]).result
        """
    )

    inspector_result = (
        PipelineInspector.on_pipeline_from_string(test_code)
        .set_code_reference_tracking(True)
        .execute()
    )

    filter_dag_for_nodes_with_ids(
        inspector_result, {5, 6, 7, 8, 9, 10, 11, 12}, 13
    )

    expected_dag = networkx.DiGraph()
    expected_classifier = DagNode(
        5,
        BasicCodeLocation("<string-source>", 26),
        OperatorContext(
            OperatorType.ESTIMATOR,
            FunctionInfo("scikeras.wrappers.KerasClassifier", "fit"),
        ),
        DagNodeDetails("Neural Network", []),
        OptionalCodeInfo(
            CodeReference(26, 6, 26, 118),
            "KerasClassifier(model=create_model, epochs=15, batch_size=1, "
            "verbose=0, input_dim=2, loss='binary_crossentropy')",
        ),
    )
    expected_explainer_creation = DagNode(
        6,
        BasicCodeLocation("<string-source>", 29),
        OperatorContext(
            OperatorType.CREATE_EXPLAINER,
            FunctionInfo("dalex.Explainer", "__init__"),
        ),
        DagNodeDetails("DALEX Explainer", []),
        OptionalCodeInfo(
            CodeReference(29, 18, 31, 1),
            "dalex.Explainer(\n    clf, data=train, y=labels, predict_function=KerasClassifier.predict\n)",
        ),
    )
    expected_test_data_explainer_creation = DagNode(
        7,
        BasicCodeLocation("<string-source>", 29),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("dalex.Explainer", "__init__"),
        ),
        DagNodeDetails(None, ["array"]),
        OptionalCodeInfo(
            CodeReference(29, 18, 31, 1),
            "dalex.Explainer(\n    clf, data=train, y=labels, predict_function=KerasClassifier.predict\n)",
        ),
    )
    expected_test_labels_explainer_creation = DagNode(
        8,
        BasicCodeLocation("<string-source>", 29),
        OperatorContext(
            OperatorType.TEST_LABELS,
            FunctionInfo("dalex.Explainer", "__init__"),
        ),
        DagNodeDetails(None, ["array"]),
        OptionalCodeInfo(
            CodeReference(29, 18, 31, 1),
            "dalex.Explainer(\n    clf, data=train, y=labels, predict_function=KerasClassifier.predict\n)",
        ),
    )
    expected_explainability_model_parts = DagNode(
        9,
        BasicCodeLocation("<string-source>", 32),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("dalex.Explainer", "model_parts"),
        ),
        DagNodeDetails("variable_importance", columns=[]),
        OptionalCodeInfo(
            CodeReference(32, 14, 32, 43),
            "dalex_explainer.model_parts()",
        ),
    )
    expected_data_source_predict_parts = DagNode(
        10,
        BasicCodeLocation("<string-source>", 34),
        OperatorContext(
            OperatorType.DATA_SOURCE,
            FunctionInfo("pandas.core.frame", "DataFrame"),
        ),
        DagNodeDetails(None, columns=[0, 1]),
        OptionalCodeInfo(
            CodeReference(34, 5, 34, 50),
            'pd.DataFrame([train[0]], index=["first_row"])',
        ),
    )
    expected_test_data_explainability_predict_parts = DagNode(
        11,
        BasicCodeLocation("<string-source>", 35),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("dalex.Explainer", "predict_parts"),
        ),
        DagNodeDetails(None, columns=[0, 1]),
        OptionalCodeInfo(
            CodeReference(35, 19, 35, 71),
            "dalex_explainer.predict_parts(df, label=df.index[0])",
        ),
    )
    expected_explainability_predict_parts = DagNode(
        12,
        BasicCodeLocation("<string-source>", 35),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("dalex.Explainer", "predict_parts"),
        ),
        DagNodeDetails("break_down_interactions", []),
        OptionalCodeInfo(
            CodeReference(35, 19, 35, 71),
            "dalex_explainer.predict_parts(df, label=df.index[0])",
        ),
    )
    expected_dag.add_edge(expected_classifier, expected_explainer_creation)
    expected_dag.add_edge(
        expected_test_data_explainer_creation, expected_explainer_creation
    )
    expected_dag.add_edge(
        expected_test_labels_explainer_creation, expected_explainer_creation
    )
    expected_dag.add_edge(
        expected_explainer_creation, expected_explainability_model_parts
    )
    expected_dag.add_edge(
        expected_explainer_creation, expected_explainability_predict_parts
    )
    expected_dag.add_edge(
        expected_data_source_predict_parts,
        expected_test_data_explainability_predict_parts,
    )
    expected_dag.add_edge(
        expected_test_data_explainability_predict_parts,
        expected_explainability_predict_parts,
    )

    compare(
        networkx.to_dict_of_dicts(inspector_result.dag),
        networkx.to_dict_of_dicts(expected_dag),
    )


def test_dalex_explainer_sgd_classifier() -> None:
    test_code = cleandoc(
        """
                import pandas as pd
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import SGDClassifier
                import numpy as np
                import dalex

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3]})
                labels = np.array([0, 0, 1, 1])

                train = StandardScaler().fit_transform(df)

                clf = SGDClassifier(loss="log_loss", random_state=42)
                clf = clf.fit(train, labels)

                dalex_explainer = dalex.Explainer(
                    clf, data=train, y=labels, predict_function=SGDClassifier.predict
                )
                explanation = dalex_explainer.model_parts()
                train_explanation = explanation.result
                df = pd.DataFrame([train[0]], index=["first_row"])
                test_explanation = dalex_explainer.predict_parts(df, label=df.index[0]).result
        """
    )

    inspector_result = (
        PipelineInspector.on_pipeline_from_string(test_code)
        .set_code_reference_tracking(True)
        .execute()
    )

    filter_dag_for_nodes_with_ids(
        inspector_result, {5, 6, 7, 8, 9, 10, 11, 12}, 13
    )

    expected_dag = networkx.DiGraph()
    expected_classifier = DagNode(
        5,
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
        6,
        BasicCodeLocation("<string-source>", 15),
        OperatorContext(
            OperatorType.CREATE_EXPLAINER,
            FunctionInfo("dalex.Explainer", "__init__"),
        ),
        DagNodeDetails("DALEX Explainer", []),
        OptionalCodeInfo(
            CodeReference(15, 18, 17, 1),
            "dalex.Explainer(\n    clf, data=train, y=labels, predict_function=SGDClassifier.predict\n)",
        ),
    )
    expected_test_data_explainer_creation = DagNode(
        7,
        BasicCodeLocation("<string-source>", 15),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("dalex.Explainer", "__init__"),
        ),
        DagNodeDetails(None, ["array"]),
        OptionalCodeInfo(
            CodeReference(15, 18, 17, 1),
            "dalex.Explainer(\n    clf, data=train, y=labels, predict_function=SGDClassifier.predict\n)",
        ),
    )
    expected_test_labels_explainer_creation = DagNode(
        8,
        BasicCodeLocation("<string-source>", 15),
        OperatorContext(
            OperatorType.TEST_LABELS,
            FunctionInfo("dalex.Explainer", "__init__"),
        ),
        DagNodeDetails(None, ["array"]),
        OptionalCodeInfo(
            CodeReference(15, 18, 17, 1),
            "dalex.Explainer(\n    clf, data=train, y=labels, predict_function=SGDClassifier.predict\n)",
        ),
    )
    expected_explainability_model_parts = DagNode(
        9,
        BasicCodeLocation("<string-source>", 18),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("dalex.Explainer", "model_parts"),
        ),
        DagNodeDetails("variable_importance", columns=[]),
        OptionalCodeInfo(
            CodeReference(18, 14, 18, 43),
            "dalex_explainer.model_parts()",
        ),
    )
    expected_data_source_predict_parts = DagNode(
        10,
        BasicCodeLocation("<string-source>", 20),
        OperatorContext(
            OperatorType.DATA_SOURCE,
            FunctionInfo("pandas.core.frame", "DataFrame"),
        ),
        DagNodeDetails(None, columns=[0, 1]),
        OptionalCodeInfo(
            CodeReference(20, 5, 20, 50),
            'pd.DataFrame([train[0]], index=["first_row"])',
        ),
    )
    expected_test_data_explainability_predict_parts = DagNode(
        11,
        BasicCodeLocation("<string-source>", 21),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("dalex.Explainer", "predict_parts"),
        ),
        DagNodeDetails(None, columns=[0, 1]),
        OptionalCodeInfo(
            CodeReference(21, 19, 21, 71),
            "dalex_explainer.predict_parts(df, label=df.index[0])",
        ),
    )
    expected_explainability_predict_parts = DagNode(
        12,
        BasicCodeLocation("<string-source>", 21),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("dalex.Explainer", "predict_parts"),
        ),
        DagNodeDetails("break_down_interactions", []),
        OptionalCodeInfo(
            CodeReference(21, 19, 21, 71),
            "dalex_explainer.predict_parts(df, label=df.index[0])",
        ),
    )
    expected_dag.add_edge(expected_classifier, expected_explainer_creation)
    expected_dag.add_edge(
        expected_test_data_explainer_creation, expected_explainer_creation
    )
    expected_dag.add_edge(
        expected_test_labels_explainer_creation, expected_explainer_creation
    )
    expected_dag.add_edge(
        expected_explainer_creation, expected_explainability_model_parts
    )
    expected_dag.add_edge(
        expected_explainer_creation, expected_explainability_predict_parts
    )
    expected_dag.add_edge(
        expected_data_source_predict_parts,
        expected_test_data_explainability_predict_parts,
    )
    expected_dag.add_edge(
        expected_test_data_explainability_predict_parts,
        expected_explainability_predict_parts,
    )

    compare(
        networkx.to_dict_of_dicts(inspector_result.dag),
        networkx.to_dict_of_dicts(expected_dag),
    )


def test_dalex_explainer_decision_tree_classifier() -> None:
    test_code = cleandoc(
        """
                import pandas as pd
                from sklearn.preprocessing import StandardScaler
                from sklearn.tree import DecisionTreeClassifier
                import numpy as np
                import dalex

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3]})
                labels = np.array([0, 0, 1, 1])

                train = StandardScaler().fit_transform(df)

                clf = DecisionTreeClassifier()
                clf = clf.fit(train, labels)

                dalex_explainer = dalex.Explainer(
                    clf, data=train, y=labels, predict_function=DecisionTreeClassifier.predict
                )
                explanation = dalex_explainer.model_parts()
                train_explanation = explanation.result
                df = pd.DataFrame([train[0]], index=["first_row"])
                test_explanation = dalex_explainer.predict_parts(df, label=df.index[0]).result
        """
    )

    inspector_result = (
        PipelineInspector.on_pipeline_from_string(test_code)
        .set_code_reference_tracking(True)
        .execute()
    )

    filter_dag_for_nodes_with_ids(
        inspector_result, {5, 6, 7, 8, 9, 10, 11, 12}, 13
    )

    expected_dag = networkx.DiGraph()
    expected_classifier = DagNode(
        5,
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
        6,
        BasicCodeLocation("<string-source>", 15),
        OperatorContext(
            OperatorType.CREATE_EXPLAINER,
            FunctionInfo("dalex.Explainer", "__init__"),
        ),
        DagNodeDetails("DALEX Explainer", []),
        OptionalCodeInfo(
            CodeReference(15, 18, 17, 1),
            "dalex.Explainer(\n    clf, data=train, y=labels, predict_function=DecisionTreeClassifier.predict\n)",
        ),
    )
    expected_test_data_explainer_creation = DagNode(
        7,
        BasicCodeLocation("<string-source>", 15),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("dalex.Explainer", "__init__"),
        ),
        DagNodeDetails(None, ["array"]),
        OptionalCodeInfo(
            CodeReference(15, 18, 17, 1),
            "dalex.Explainer(\n    clf, data=train, y=labels, predict_function=DecisionTreeClassifier.predict\n)",
        ),
    )
    expected_test_labels_explainer_creation = DagNode(
        8,
        BasicCodeLocation("<string-source>", 15),
        OperatorContext(
            OperatorType.TEST_LABELS,
            FunctionInfo("dalex.Explainer", "__init__"),
        ),
        DagNodeDetails(None, ["array"]),
        OptionalCodeInfo(
            CodeReference(15, 18, 17, 1),
            "dalex.Explainer(\n    clf, data=train, y=labels, predict_function=DecisionTreeClassifier.predict\n)",
        ),
    )
    expected_explainability_model_parts = DagNode(
        9,
        BasicCodeLocation("<string-source>", 18),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("dalex.Explainer", "model_parts"),
        ),
        DagNodeDetails("variable_importance", columns=[]),
        OptionalCodeInfo(
            CodeReference(18, 14, 18, 43),
            "dalex_explainer.model_parts()",
        ),
    )
    expected_data_source_predict_parts = DagNode(
        10,
        BasicCodeLocation("<string-source>", 20),
        OperatorContext(
            OperatorType.DATA_SOURCE,
            FunctionInfo("pandas.core.frame", "DataFrame"),
        ),
        DagNodeDetails(None, columns=[0, 1]),
        OptionalCodeInfo(
            CodeReference(20, 5, 20, 50),
            'pd.DataFrame([train[0]], index=["first_row"])',
        ),
    )
    expected_test_data_explainability_predict_parts = DagNode(
        11,
        BasicCodeLocation("<string-source>", 21),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("dalex.Explainer", "predict_parts"),
        ),
        DagNodeDetails(None, columns=[0, 1]),
        OptionalCodeInfo(
            CodeReference(21, 19, 21, 71),
            "dalex_explainer.predict_parts(df, label=df.index[0])",
        ),
    )
    expected_explainability_predict_parts = DagNode(
        12,
        BasicCodeLocation("<string-source>", 21),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("dalex.Explainer", "predict_parts"),
        ),
        DagNodeDetails("break_down_interactions", []),
        OptionalCodeInfo(
            CodeReference(21, 19, 21, 71),
            "dalex_explainer.predict_parts(df, label=df.index[0])",
        ),
    )
    expected_dag.add_edge(expected_classifier, expected_explainer_creation)
    expected_dag.add_edge(
        expected_test_data_explainer_creation, expected_explainer_creation
    )
    expected_dag.add_edge(
        expected_test_labels_explainer_creation, expected_explainer_creation
    )
    expected_dag.add_edge(
        expected_explainer_creation, expected_explainability_model_parts
    )
    expected_dag.add_edge(
        expected_explainer_creation, expected_explainability_predict_parts
    )
    expected_dag.add_edge(
        expected_data_source_predict_parts,
        expected_test_data_explainability_predict_parts,
    )
    expected_dag.add_edge(
        expected_test_data_explainability_predict_parts,
        expected_explainability_predict_parts,
    )

    compare(
        networkx.to_dict_of_dicts(inspector_result.dag),
        networkx.to_dict_of_dicts(expected_dag),
    )


def test_dalex_explainer_logistic_regression() -> None:
    test_code = cleandoc(
        """
                import pandas as pd
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                import numpy as np
                import dalex

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3]})
                labels = np.array([0, 0, 1, 1])

                train = StandardScaler().fit_transform(df)

                clf = LogisticRegression()
                clf = clf.fit(train, labels)

                dalex_explainer = dalex.Explainer(
                    clf, data=train, y=labels, predict_function=LogisticRegression.predict
                )
                explanation = dalex_explainer.model_parts()
                train_explanation = explanation.result
                df = pd.DataFrame([train[0]], index=["first_row"])
                test_explanation = dalex_explainer.predict_parts(df, label=df.index[0]).result
        """
    )

    inspector_result = (
        PipelineInspector.on_pipeline_from_string(test_code)
        .set_code_reference_tracking(True)
        .execute()
    )

    filter_dag_for_nodes_with_ids(
        inspector_result, {5, 6, 7, 8, 9, 10, 11, 12}, 13
    )

    expected_dag = networkx.DiGraph()
    expected_classifier = DagNode(
        5,
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
        6,
        BasicCodeLocation("<string-source>", 15),
        OperatorContext(
            OperatorType.CREATE_EXPLAINER,
            FunctionInfo("dalex.Explainer", "__init__"),
        ),
        DagNodeDetails("DALEX Explainer", []),
        OptionalCodeInfo(
            CodeReference(15, 18, 17, 1),
            "dalex.Explainer(\n    clf, data=train, y=labels, predict_function=LogisticRegression.predict\n)",
        ),
    )
    expected_test_data_explainer_creation = DagNode(
        7,
        BasicCodeLocation("<string-source>", 15),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("dalex.Explainer", "__init__"),
        ),
        DagNodeDetails(None, ["array"]),
        OptionalCodeInfo(
            CodeReference(15, 18, 17, 1),
            "dalex.Explainer(\n    clf, data=train, y=labels, predict_function=LogisticRegression.predict\n)",
        ),
    )
    expected_test_labels_explainer_creation = DagNode(
        8,
        BasicCodeLocation("<string-source>", 15),
        OperatorContext(
            OperatorType.TEST_LABELS,
            FunctionInfo("dalex.Explainer", "__init__"),
        ),
        DagNodeDetails(None, ["array"]),
        OptionalCodeInfo(
            CodeReference(15, 18, 17, 1),
            "dalex.Explainer(\n    clf, data=train, y=labels, predict_function=LogisticRegression.predict\n)",
        ),
    )
    expected_explainability_model_parts = DagNode(
        9,
        BasicCodeLocation("<string-source>", 18),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("dalex.Explainer", "model_parts"),
        ),
        DagNodeDetails("variable_importance", columns=[]),
        OptionalCodeInfo(
            CodeReference(18, 14, 18, 43),
            "dalex_explainer.model_parts()",
        ),
    )
    expected_data_source_predict_parts = DagNode(
        10,
        BasicCodeLocation("<string-source>", 20),
        OperatorContext(
            OperatorType.DATA_SOURCE,
            FunctionInfo("pandas.core.frame", "DataFrame"),
        ),
        DagNodeDetails(None, columns=[0, 1]),
        OptionalCodeInfo(
            CodeReference(20, 5, 20, 50),
            'pd.DataFrame([train[0]], index=["first_row"])',
        ),
    )
    expected_test_data_explainability_predict_parts = DagNode(
        11,
        BasicCodeLocation("<string-source>", 21),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("dalex.Explainer", "predict_parts"),
        ),
        DagNodeDetails(None, columns=[0, 1]),
        OptionalCodeInfo(
            CodeReference(21, 19, 21, 71),
            "dalex_explainer.predict_parts(df, label=df.index[0])",
        ),
    )
    expected_explainability_predict_parts = DagNode(
        12,
        BasicCodeLocation("<string-source>", 21),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("dalex.Explainer", "predict_parts"),
        ),
        DagNodeDetails("break_down_interactions", []),
        OptionalCodeInfo(
            CodeReference(21, 19, 21, 71),
            "dalex_explainer.predict_parts(df, label=df.index[0])",
        ),
    )
    expected_dag.add_edge(expected_classifier, expected_explainer_creation)
    expected_dag.add_edge(
        expected_test_data_explainer_creation, expected_explainer_creation
    )
    expected_dag.add_edge(
        expected_test_labels_explainer_creation, expected_explainer_creation
    )
    expected_dag.add_edge(
        expected_explainer_creation, expected_explainability_model_parts
    )
    expected_dag.add_edge(
        expected_explainer_creation, expected_explainability_predict_parts
    )
    expected_dag.add_edge(
        expected_data_source_predict_parts,
        expected_test_data_explainability_predict_parts,
    )
    expected_dag.add_edge(
        expected_test_data_explainability_predict_parts,
        expected_explainability_predict_parts,
    )

    compare(
        networkx.to_dict_of_dicts(inspector_result.dag),
        networkx.to_dict_of_dicts(expected_dag),
    )
