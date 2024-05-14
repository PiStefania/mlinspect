from typing import List

from mlinspect import InspectorResult


def filter_dag_for_nodes_with_ids(
    inspector_result: InspectorResult,
    node_ids: List[int] | set[int],
    total_expected_node_num: int,
) -> None:
    """
    Filter for DAG Nodes relevant for this test
    """
    assert len(inspector_result.dag.nodes) == total_expected_node_num
    dag_nodes_irrelevant__for_test = [
        dag_node
        for dag_node in list(inspector_result.dag.nodes)
        if dag_node.node_id not in node_ids
    ]
    inspector_result.dag.remove_nodes_from(dag_nodes_irrelevant__for_test)
    assert len(inspector_result.dag.nodes) == len(node_ids)
