Explainability Methods - Dag Representation
==============================================

## Methods
The following methods/packages are available for DAG representation:
- `SHAP`
  - `KernelExplainer` with `shap_values`
- `LIME`
  - `LimeTabularExplainer` with `explain_instance`
- `PDP`
  - `PartialDependenceDisplay` with `from_estimator`
- `ICE`
  - `PartialDependenceDisplay` with `from_estimator` using kind=`individual`
- `IG`
  - `IntegratedGradients` with `explain`
- `ALE`
  - `ALE` with `explain`
- `DALE`
  - `DALE` with `eval`
- `DALEX`
  - `Explainer` with `model_parts` and `predict_parts`

## Implementation
The explainability methods have been patched using the `gorilla` package. Dag nodes are created for each patched method and the results can be viewed under each respective module.

## Example
An example of how to use the DAG representation is shown below:
```python
from features.explainability.monkey_patching import patch_shap, patch_lime, patch_sklearn_inspection, patch_alibi, \
    patch_dale, patch_dalex
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Some tensorflow warnings in the pipeline we inspect
from mlinspect.utils import get_project_root

from mlinspect import PipelineInspector

EXAMPLE_PIPELINE = os.path.join(str(get_project_root()), "features", "explainability", "pipeline_with_dag.py")

inspector_result = PipelineInspector\
    .on_pipeline_from_py_file(EXAMPLE_PIPELINE) \
    .add_custom_monkey_patching_module(patch_shap) \
    .add_custom_monkey_patching_module(patch_lime) \
    .add_custom_monkey_patching_module(patch_sklearn_inspection) \
    .add_custom_monkey_patching_module(patch_alibi) \
    .add_custom_monkey_patching_module(patch_dale) \
    .add_custom_monkey_patching_module(patch_dalex) \
    .execute()

extracted_dag = inspector_result.dag
dag_node_to_inspection_results = inspector_result.dag_node_to_inspection_results
check_results = inspector_result.check_to_check_results

from IPython.display import Image
from mlinspect.visualisation import save_fig_to_path

filename = os.path.join(str(get_project_root()), "features", "explainability", "explainability.png")
save_fig_to_path(extracted_dag, filename)

Image(filename=filename)
```

## How to extend
To extend and therefore patch additional explainability methods, just add a new patching function to the `features.explainability.monkey_patching` module. The patching function should be able to handle the new method. Just check the existing patching functions for reference. Pay extra attention to the `gorilla` package documentation to understand how to patch the new method and also keep in mind the different inputs and output that a dag node has.

## Notes
Most of the explainability methods are coupled with the corresponding model prediction methods. Some patched model prediction functions need to be bypassed in order to omit additional unnecessary dag nodes being created.
