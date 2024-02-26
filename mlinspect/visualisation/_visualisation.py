"""
Utility functions to visualise the extracted DAG
"""

from inspect import cleandoc

import networkx
from networkx import DiGraph
from networkx.drawing.nx_agraph import to_agraph

from mlinspect import DagNode


def save_fig_to_path(extracted_dag: DiGraph, filename: str) -> None:
    """
    Create a figure of the extracted DAG and save it with some filename
    """

    def get_new_node_label(node: DagNode) -> str:
        label = cleandoc(
            f"""
                {node.node_id}: {node.operator_info.operator.value} (L{node.code_location.lineno})
                {node.details.description or ""}
                """
        )
        return label

    # noinspection PyTypeChecker
    extracted_dag = networkx.relabel_nodes(extracted_dag, get_new_node_label)

    agraph = to_agraph(extracted_dag)
    agraph.layout("dot")
    agraph.draw(filename)


def get_dag_as_pretty_string(extracted_dag: DiGraph) -> str:
    """
    Create a figure of the extracted DAG and save it with some filename
    """

    def get_new_node_label(node: DagNode) -> str:
        description = ""
        if node.details.description:
            description = "({})".format(node.details.description)

        label = "{}{}".format(node.operator_info.operator.value, description)
        return label

    # noinspection PyTypeChecker
    extracted_dag = networkx.relabel_nodes(extracted_dag, get_new_node_label)

    agraph = to_agraph(extracted_dag)
    graph_str: str = agraph.to_string()
    return graph_str
