import os

from mlinspect.utils import get_project_root

COMPAS_DECISION_TREE_CLASSIFIER_PY = os.path.join(
    str(get_project_root()),
    "features",
    "explainability",
    "pipelines",
    "compas_decision_tree_classifier.py",
)

COMPAS_SGD_CLASSIFIER_PY = os.path.join(
    str(get_project_root()),
    "features",
    "explainability",
    "pipelines",
    "compas_sgd_classifier.py",
)

EXPLAINABILITY_LIME_PY = os.path.join(
    str(get_project_root()),
    "features",
    "explainability",
    "pipelines",
    "adult_simple_with_explainability.py",
)
