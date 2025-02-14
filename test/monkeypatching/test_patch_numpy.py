"""
Tests whether the monkey patching works for all patched numpy methods
"""

from inspect import cleandoc

import pandas
from pandas import DataFrame
from testfixtures import compare

from mlinspect import FunctionInfo, OperatorContext, OperatorType
from mlinspect.inspections._lineage import LineageId, RowLineage
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import (
    BasicCodeLocation,
    CodeReference,
    DagNode,
    DagNodeDetails,
    OptionalCodeInfo,
)


def test_numpy_random() -> None:
    """
    Tests whether the monkey patching of ('numpy.random', 'random') works
    """
    test_code = cleandoc(
        """
            import numpy as np
            np.random.seed(42)
            test = np.random.random(100)
            assert len(test) == 100
        """
    )

    inspector_result = _pipeline_executor.singleton.run(
        python_code=test_code,
        track_code_references=True,
        inspections=[RowLineage(2)],
    )
    extracted_node: DagNode = list(inspector_result.dag.nodes)[0]

    expected_node = DagNode(
        0,
        BasicCodeLocation("<string-source>", 3),
        OperatorContext(
            OperatorType.DATA_SOURCE, FunctionInfo("numpy.random", "random")
        ),
        DagNodeDetails("random", ["array"]),
        OptionalCodeInfo(CodeReference(3, 7, 3, 28), "np.random.random(100)"),
    )
    compare(extracted_node, expected_node)

    inspection_results_data_source = (
        inspector_result.dag_node_to_inspection_results[extracted_node]
    )
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame(
        [[0.5, {LineageId(0, 0)}], [0.5, {LineageId(0, 1)}]],
        columns=["array", "mlinspect_lineage"],
    )
    pandas.testing.assert_frame_equal(
        lineage_output.reset_index(drop=True),
        expected_lineage_df.reset_index(drop=True),
        atol=1,
    )


def test_numpy_array() -> None:
    """
    Tests whether the monkey patching of ('numpy', 'array') works
    """
    test_code = cleandoc(
        """
            import numpy as np
            test = np.array([0, 1])
        """
    )

    inspector_result = _pipeline_executor.singleton.run(
        python_code=test_code,
        track_code_references=True,
        inspections=[RowLineage(2)],
    )
    extracted_node: DagNode = list(inspector_result.dag.nodes)[0]

    expected_node = DagNode(
        0,
        BasicCodeLocation("<string-source>", 2),
        OperatorContext(
            OperatorType.DATA_SOURCE, FunctionInfo("numpy", "array")
        ),
        DagNodeDetails("array", ["array"]),
        OptionalCodeInfo(CodeReference(2, 7, 2, 23), "np.array([0, 1])"),
    )
    compare(extracted_node, expected_node)

    inspection_results_data_source = (
        inspector_result.dag_node_to_inspection_results[extracted_node]
    )
    lineage_output = inspection_results_data_source[RowLineage(2)]
    expected_lineage_df = DataFrame(
        [[0, {LineageId(0, 0)}], [1, {LineageId(0, 1)}]],
        columns=["array", "mlinspect_lineage"],
    )
    pandas.testing.assert_frame_equal(
        lineage_output.reset_index(drop=True),
        expected_lineage_df.reset_index(drop=True),
        atol=1,
    )
