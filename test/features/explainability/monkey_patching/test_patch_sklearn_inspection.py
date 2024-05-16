from inspect import cleandoc

import networkx
from testfixtures import compare

from features.explainability.monkey_patching import patch_sklearn_inspection

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


def test_sklearn_inspection_pdp_from_estimator() -> None:
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
                from sklearn.inspection import PartialDependenceDisplay
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

                display_pdp = PartialDependenceDisplay.from_estimator(
                    estimator=clf, X=train, features=[0, 1], kind="average"
                )
        """
    )

    inspector_result = (
        PipelineInspector.on_pipeline_from_string(test_code)
        .set_code_reference_tracking(True)
        .add_custom_monkey_patching_module(patch_sklearn_inspection)
        .execute()
    )

    filter_dag_for_nodes_with_ids(inspector_result, {7, 8, 9}, 10)

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
    expected_test_data_explainability = DagNode(
        8,
        BasicCodeLocation("<string-source>", 29),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("PartialDependenceDisplay", "from_estimator"),
        ),
        DagNodeDetails(None, columns=["array"]),
        OptionalCodeInfo(
            CodeReference(29, 14, 31, 1),
            'PartialDependenceDisplay.from_estimator(\n    estimator=clf, X=train, features=[0, 1], kind="average"\n)',
        ),
    )
    expected_explainability = DagNode(
        9,
        BasicCodeLocation("<string-source>", 29),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("PartialDependenceDisplay", "from_estimator"),
        ),
        DagNodeDetails("PDP", []),
        OptionalCodeInfo(
            CodeReference(29, 14, 31, 1),
            'PartialDependenceDisplay.from_estimator(\n    estimator=clf, X=train, features=[0, 1], kind="average"\n)',
        ),
    )
    expected_dag.add_edge(expected_classifier, expected_explainability)
    expected_dag.add_edge(
        expected_test_data_explainability, expected_explainability
    )

    compare(
        networkx.to_dict_of_dicts(inspector_result.dag),
        networkx.to_dict_of_dicts(expected_dag),
    )


def test_sklearn_inspection_ice_from_estimator() -> None:
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
                from sklearn.inspection import PartialDependenceDisplay
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

                display_pdp = PartialDependenceDisplay.from_estimator(
                    estimator=clf, X=train, features=[0, 1], kind="individual"
                )
        """
    )

    inspector_result = (
        PipelineInspector.on_pipeline_from_string(test_code)
        .set_code_reference_tracking(True)
        .add_custom_monkey_patching_module(patch_sklearn_inspection)
        .execute()
    )

    filter_dag_for_nodes_with_ids(inspector_result, {7, 8, 9}, 10)

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
    expected_test_data_explainability = DagNode(
        8,
        BasicCodeLocation("<string-source>", 29),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("PartialDependenceDisplay", "from_estimator"),
        ),
        DagNodeDetails(None, columns=["array"]),
        OptionalCodeInfo(
            CodeReference(29, 14, 31, 1),
            'PartialDependenceDisplay.from_estimator(\n    estimator=clf, X=train, features=[0, 1], kind="individual"\n)',
        ),
    )
    expected_explainability = DagNode(
        9,
        BasicCodeLocation("<string-source>", 29),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("PartialDependenceDisplay", "from_estimator"),
        ),
        DagNodeDetails("ICE", []),
        OptionalCodeInfo(
            CodeReference(29, 14, 31, 1),
            'PartialDependenceDisplay.from_estimator(\n    estimator=clf, X=train, features=[0, 1], kind="individual"\n)',
        ),
    )
    expected_dag.add_edge(expected_classifier, expected_explainability)
    expected_dag.add_edge(
        expected_test_data_explainability, expected_explainability
    )

    compare(
        networkx.to_dict_of_dicts(inspector_result.dag),
        networkx.to_dict_of_dicts(expected_dag),
    )
