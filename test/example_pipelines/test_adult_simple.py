"""
Tests whether the adult_easy test pipeline works
"""

import ast

import nbformat
from nbconvert import PythonExporter

from example_pipelines import (
    ADULT_SIMPLE_IPYNB,
    ADULT_SIMPLE_PNG,
    ADULT_SIMPLE_PY,
)

from mlinspect.testing._testing_helper_utils import (
    run_and_assert_all_op_outputs_inspected,
)


def test_py_pipeline_runs() -> None:
    """
    Tests whether the .py version of the pipeline works
    """
    with open(ADULT_SIMPLE_PY, encoding="utf-8") as file:
        text = file.read()
        parsed_ast = ast.parse(text)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"))


def test_nb_pipeline_runs() -> None:
    """
    Tests whether the .ipynb version of the pipeline works
    """
    with open(ADULT_SIMPLE_IPYNB, encoding="utf-8") as file:
        notebook = nbformat.reads(file.read(), nbformat.NO_CONVERT)
        exporter = PythonExporter()

        code, _ = exporter.from_notebook_node(notebook)
        parsed_ast = ast.parse(code)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"))


def test_instrumented_py_pipeline_runs() -> None:
    """
    Tests whether the pipeline works with instrumentation
    """
    dag = run_and_assert_all_op_outputs_inspected(
        ADULT_SIMPLE_PY, ["race"], ADULT_SIMPLE_PNG
    )
    assert len(dag) == 12
