### Purpose
This project provide python code to train, predict, and explain the risk of failure for project

### Requirements:
You need to install the following requirement packages
- shap 
- hpsklearn

### Install the package
pip install -i https://test.pypi.org/simple/ riskfailure-gainx>=0.0.3

### Define the risk failure models
The package provide two classes for specifying and training a risk of failure model

    (a) IntraRiskFailureModels: This class defines proprieties of risk failure models in intra universes. It also provides methods to predict and explain the risk of failure 
    (b) ParallelRiskFailureModels: This class defines proprieties of risk failure models in parallel universes. It also provides methods to predict and explain the risk of failure 

model_names=['end_date_failure', 'budget_failure', 'benefit_failure','project_failure']
intra_models=IntraRiskFailureModels(model_names=model_names, max_evals=100, n_repeats=50)
parallel_models=ParallelRiskFailureModels(model_names=model_names, max_evals=100, n_repeats=50)

### Train the model
x: features
y1: end date failure
y2: budget failure
y3: benefit failure
y4: project failure = y1 U y2 U y3

intra_models.fit(X, y1, y2, y3)
parallel_models.fit(X, y1, y2, y3, y4)

### Save the trained models
model_path=output_path + '/models'
rmtree(model_path, ignore_errors=True)
mkdir(model_path)
write_pickle_file(model_path,intra_models,'intra_models')
write_pickle_file(model_path,parallel_models,'parallel_models')


### Load models: intra risk of failure
model_path=output_path + "/models"
intra_models=read_pickle_file(model_path, 'intra_models')

### Predict the probability of failure
project_features=features.values.astype('float32')
proba=intra_models.get_proba(project_features)

### Predict shapley values 
project_features=features
shap_values=intra_models.get_shap_values(project_features)

### Display shapley values for a given project
intra_models.display_local_shap_values(proj_id=2, max_display=10)

### Display shapley values for all project
intra_models.display_global_shap_values(max_display=10)


### More Demo for the package 
For more information on how to use the package, visit the Bibucket repository at
https://bitbucket.org/gainx/generalresearch/src/master/predicted_to_fail/risk_failure/













