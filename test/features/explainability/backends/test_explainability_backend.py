from features.explainability.pipelines import EXPLAINABILITY_LIME_PY

from mlinspect.testing._testing_helper_utils import (
    run_multiple_test_analyzers,
    run_random_annotation_testing_analyzer,
    run_row_index_annotation_testing_analyzer,
)


def test_explainability_backend_random_annotation_propagation() -> None:
    """
    Tests whether the pandas backend works
    """
    with open(EXPLAINABILITY_LIME_PY, encoding="utf-8") as file:
        code = file.read()
        random_annotation_analyzer_result = (
            run_random_annotation_testing_analyzer(code)
        )
        assert len(random_annotation_analyzer_result) == 25


def test_explainability_backend_row_index_annotation_propagation() -> None:
    """
    Tests whether the pandas backend works
    """
    with open(EXPLAINABILITY_LIME_PY, encoding="utf-8") as file:
        code = file.read()
        lineage_result = run_row_index_annotation_testing_analyzer(code)
        assert len(lineage_result) == 25


def test_explainability_backend_annotation_propagation_multiple_analyzers() -> (
    None
):
    """
    Tests whether the pandas backend works
    """
    with open(EXPLAINABILITY_LIME_PY, encoding="utf-8") as file:
        code = file.read()

        dag_node_to_inspection_results, analyzers = (
            run_multiple_test_analyzers(code)
        )

        for inspection_result in dag_node_to_inspection_results.values():
            for analyzer in analyzers:
                assert analyzer in inspection_result
