Explainability Methods - Inspection
==============================================

## Methods
The following methods/packages are available for inspection:
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
The explainability methods have been implemented in an inspection that is named `Explainer`. The inspection is capable to calculate the explainability metrics for the given methods after the model fitting. It then uses the model prediction to calculate the explainability metrics. Then, one can just check the results of the explainability methods by using the `inspector_result` object. 

## How to use
Just add the desired inspection `Explainer` to the `PipelineInspector` object and execute the pipeline. The results will be available in the `inspector_result` object.

### Inputs
- `ExplainabilityMethodsEnum` - Enum containing the available methods for inspection
- `Input data` - The input data for the inspection
- `Test data` - The test data for the inspection, to use for predicting explainability
- `Feature names` - The feature names for the explainability
- `Test labels` - The test labels for the explainability metric, if needed
- `Train labels` - The train labels for the explainability metric, if needed
- `nsamples` - The number of samples to use for the explainability metric, if needed

```python
from features.explainability.inspections.explainability_methods_enum import ExplainabilityMethodsEnum
from features.explainability.inspections.explainer import Explainer
from mlinspect import PipelineInspector

inspector_result = PipelineInspector\
    .on_pipeline_from_py_file(EXAMPLE_PIPELINE) \
    .add_required_inspection(Explainer([ExplainabilityMethodsEnum.SHAP, ExplainabilityMethodsEnum.LIME, ExplainabilityMethodsEnum.PDP, ExplainabilityMethodsEnum.ICE, ExplainabilityMethodsEnum.INTEGRATED_GRADIENTS, ExplainabilityMethodsEnum.ALE, ExplainabilityMethodsEnum.DALE, ExplainabilityMethodsEnum.DALEX], X_t_train.view(np.ndarray), X_t_test.view(np.ndarray), featurisation.get_feature_names_out(), y_test, y_train)) \
    .execute()

extracted_dag = inspector_result.dag
dag_node_to_inspection_results = inspector_result.dag_node_to_inspection_results
check_results = inspector_result.check_to_check_results
# e.g. for LIME
relevant_node = [node for node in extracted_dag.nodes if node.operator_info.operator in {OperatorType.ESTIMATOR,}][0]
inspection_result = dag_node_to_inspection_results[relevant_node][Explainer([ExplainabilityMethodsEnum.LIME,], X_t_train.view(np.ndarray), X_t_test.view(np.ndarray), featurisation.get_feature_names_out(),y_test, y_train)]
inspection_result[ExplainabilityMethodsEnum.LIME]["results"].show_in_notebook()
```

## How to extend
To extend the explainability methods, just add a new enum value to `ExplainabilityMethodsEnum` and implement the corresponding `Explainer` class that already exists. The `Explainer` class should be able to handle the new enum value and the corresponding inspection method. Just check the existing `Explainer` class for reference.
You should initiate both the explainer class and also the explanation using the test data.
