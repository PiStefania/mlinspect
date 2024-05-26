import os

from mlinspect.utils import get_project_root

EXPLAINABILITY_HEALTHCARE_PY = os.path.join(
    str(get_project_root()),
    "features",
    "explainability",
    "examples",
    "test_pipeline.py",
)
