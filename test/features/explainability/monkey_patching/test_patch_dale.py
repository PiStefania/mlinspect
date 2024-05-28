from inspect import cleandoc

import networkx
from testfixtures import compare

from features.explainability.monkey_patching import patch_dale

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


def test_dale_explainer() -> None:
    test_code = cleandoc(
        """
                import pandas as pd
                from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
                from scikeras.wrappers import KerasClassifier
                from tensorflow.keras.layers import Dense
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.optimizers.experimental import SGD
                from features.explainability.dale.dale import DALE
                import tensorflow as tf
                import numpy as np
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

                def model_grad(inp):
                    x_inp = tf.cast(inp, tf.float32)
                    with tf.GradientTape() as tape:
                        tape.watch(x_inp)
                        preds = clf.model_(x_inp)
                    grads = tape.gradient(preds, x_inp)
                    return grads.numpy()

                dale = DALE(data=train, model=clf, model_jac=model_grad)
                dale.fit()
                explanations = dale.eval(x=train[:2], s=0)
        """
    )

    inspector_result = (
        PipelineInspector.on_pipeline_from_string(test_code)
        .set_code_reference_tracking(True)
        .add_custom_monkey_patching_module(patch_dale)
        .execute()
    )

    filter_dag_for_nodes_with_ids(inspector_result, {7, 8, 9, 10, 11}, 12)

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
        BasicCodeLocation("<string-source>", 37),
        OperatorContext(
            OperatorType.CREATE_EXPLAINER,
            FunctionInfo("dale.DALE", "__init__"),
        ),
        DagNodeDetails("DALE Explainer", []),
        OptionalCodeInfo(
            CodeReference(37, 7, 37, 56),
            "DALE(data=train, model=clf, model_jac=model_grad)",
        ),
    )
    expected_test_data_explainer_creation = DagNode(
        9,
        BasicCodeLocation("<string-source>", 37),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("dale.DALE", "__init__"),
        ),
        DagNodeDetails(None, columns=["array"]),
        OptionalCodeInfo(
            CodeReference(37, 7, 37, 56),
            "DALE(data=train, model=clf, model_jac=model_grad)",
        ),
    )
    expected_test_data_explainability = DagNode(
        10,
        BasicCodeLocation("<string-source>", 39),
        OperatorContext(
            OperatorType.TEST_DATA,
            FunctionInfo("dale.DALE", "eval"),
        ),
        DagNodeDetails(None, ["array"]),
        OptionalCodeInfo(
            CodeReference(39, 15, 39, 42),
            "dale.eval(x=train[:2], s=0)",
        ),
    )
    expected_explainability = DagNode(
        11,
        BasicCodeLocation("<string-source>", 39),
        OperatorContext(
            OperatorType.EXPLAINABILITY,
            FunctionInfo("dale.DALE", "eval"),
        ),
        DagNodeDetails("DALE", []),
        OptionalCodeInfo(
            CodeReference(39, 15, 39, 42),
            "dale.eval(x=train[:2], s=0)",
        ),
    )
    expected_dag.add_edge(expected_classifier, expected_explainer_creation)
    expected_dag.add_edge(
        expected_test_data_explainer_creation, expected_explainer_creation
    )
    expected_dag.add_edge(
        expected_test_data_explainability, expected_explainability
    )
    expected_dag.add_edge(expected_explainer_creation, expected_explainability)

    compare(
        networkx.to_dict_of_dicts(inspector_result.dag),
        networkx.to_dict_of_dicts(expected_dag),
    )
