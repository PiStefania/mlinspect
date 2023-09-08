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
from example_pipelines.healthcare.healthcare_utils import MyW2VTransformer, create_model_predict
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
y = data["label"]
X = data.drop("label", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)


impute_and_one_hot_encode = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])
featurisation = ColumnTransformer(transformers=[
    ("impute_and_one_hot_encode", impute_and_one_hot_encode, ['smoker', 'county', 'race']),
    #('word2vec', MyW2VTransformer(min_count=2, vector_size=5), ['last_name']),
    ('numeric', StandardScaler(), ['num_children', 'income']),
], remainder='drop')

neural_net = KerasClassifier(model=create_model_predict, epochs=10, batch_size=1, verbose=0, loss='binary_crossentropy',)
X_t_train = featurisation.fit_transform(X_train, y_train)
X_t_test = featurisation.fit_transform(X_test, y_test)
neural_net.fit(X_t_train, y_train)
print("Mean accuracy: {}".format(neural_net.score(X_t_test, y_test)))
print(f"Predict first 10 samples: {neural_net.predict(X_t_test[:10])}")
# Introduce explainability
explainer = shap.KernelExplainer(neural_net.predict, X_t_train, feature_names=featurisation.get_feature_names_out())
shap_values = explainer.shap_values(X_t_test[:1], nsamples=100)
shap.force_plot(explainer.expected_value, shap_values, X_t_test[:1])