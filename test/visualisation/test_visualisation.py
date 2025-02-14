"""
Tests whether the visualisation of the resulting DAG works
"""

import os

from mlinspect.testing._testing_helper_utils import get_expected_dag_adult_easy
from mlinspect.utils import get_project_root
from mlinspect.visualisation import get_dag_as_pretty_string, save_fig_to_path


def test_save_fig_to_path() -> None:
    """
    Tests whether the .py version of the inspector works
    """
    extracted_dag = get_expected_dag_adult_easy("<string-source>")

    filename = os.path.join(
        str(get_project_root()),
        "example_pipelines",
        "adult_simple",
        "adult_simple.png",
    )
    save_fig_to_path(extracted_dag, filename)

    assert os.path.isfile(filename)


def test_get_dag_as_pretty_string() -> None:
    """
    Tests whether the .py version of the inspector works
    """
    extracted_dag = get_expected_dag_adult_easy("<string-source>")

    pretty_string = get_dag_as_pretty_string(extracted_dag)

    print(pretty_string)
