"""
Tests whether the adult_easy test pipeline works
"""

import ast

from example_pipelines import ADULT_COMPLEX_PNG, ADULT_COMPLEX_PY

from mlinspect.testing._testing_helper_utils import (
    run_and_assert_all_op_outputs_inspected,
)


def test_py_pipeline_runs() -> None:
    """
    Tests whether the .py version of the pipeline works
    """
    with open(ADULT_COMPLEX_PY, encoding="utf-8") as file:
        text = file.read()
        parsed_ast = ast.parse(text)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"))


def test_instrumented_py_pipeline_runs() -> None:
    """
    Tests whether the pipeline works with instrumentation
    """
    dag = run_and_assert_all_op_outputs_inspected(
        ADULT_COMPLEX_PY, ["race"], ADULT_COMPLEX_PNG
    )
    assert len(dag) == 24
