"""
Tests whether the healthcare demo works
"""

import os

import matplotlib
from importnb import Notebook

from mlinspect.utils import get_project_root

DEMO_NB_FILE = os.path.join(
    str(get_project_root()),
    "demo",
    "feature_overview",
    "feature_overview.ipynb",
)


def test_demo_nb() -> None:
    """
    Tests whether the demo notebook works
    """
    matplotlib.use(
        "template"
    )  # Disable plt.show when executing nb as part of this test
    Notebook().load_file(DEMO_NB_FILE)
