"""Predicting which patients are at a higher risk of complications"""
import warnings
import os
import pandas as pd
import shap
from scikeras.wrappers import KerasClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from example_pipelines.healthcare.healthcare_utils import create_model_predict
from mlinspect.utils import get_project_root

# FutureWarning: Sklearn 0.24 made a change that breaks remainder='drop', that change will be fixed
#  in an upcoming version: https://github.com/scikit-learn/scikit-learn/pull/19263
warnings.filterwarnings('ignore')

shap.initjs()


COUNTIES_OF_INTEREST = ['county2', 'county3']

patients = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                    "patients.csv"), na_values='?')
histories = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                     "histories.csv"), na_values='?')

data = patients.merge(histories, on=['ssn'])
complications = data.groupby('age_group') \
    .agg(mean_complications=('complications', 'mean'))
data = data.merge(complications, on=['age_group'])
data['label'] = data['complications'] > 1.2 * data['mean_complications']
data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]
data = data[data['county'].isin(COUNTIES_OF_INTEREST)]
train_data, test_data = train_test_split(data)
y_train = train_data["label"]
y_test = test_data["label"]
X_train = train_data.drop("label", axis=1)
X_test = test_data.drop("label", axis=1)

impute_and_one_hot_encode = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])
featurisation = ColumnTransformer(transformers=[
    ("impute_and_one_hot_encode", impute_and_one_hot_encode, ['smoker', 'county', 'race']),
    ('numeric', StandardScaler(), ['num_children', 'income']),
], remainder='drop')
#featurisation.set_output(transform="pandas")

neural_net = KerasClassifier(model=create_model_predict, epochs=10, batch_size=1, verbose=0, loss='binary_crossentropy',)
X_t_train = featurisation.fit_transform(train_data, y_train)
X_t_test = featurisation.fit_transform(X_test, y_test)
neural_net.fit(X_t_train, y_train)
print("Mean accuracy: {}".format(neural_net.score(X_t_test, y_test)))
print(f"Predict first 10 samples: {neural_net.predict(X_t_test[:10])}")
# Introduce explainability
# explainer = shap.KernelExplainer(neural_net.predict, X_t_train)
# shap_values = explainer.shap_values(X_t_test[:1], nsamples=100)
# print(f"{featurisation.get_feature_names_out()=}")
# shap.force_plot(explainer.expected_value, shap_values, X_t_test[:1], feature_names=featurisation.get_feature_names_out())
# shap.summary_plot(shap_values, X_t_test[:1], feature_names=featurisation.get_feature_names_out(), plot_type="bar")