"""
This module shows how to use the network_health. py module to define, fit and use the a network health model

Usage:
- python3 demo_network_health.py
"""
from shutil import rmtree
from os import mkdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from network_health import NetworkHealthModel, write_pickle_file, read_pickle_file
from network_health import generate_data


### dataset and output paths
dataset_path = "datasets"
output_path = "outputs"
model_path = output_path + "/models"

### Import data
filename = dataset_path + "/data.csv"
data = pd.read_csv(filename)



#####################################################
### Specify and train the model #####################
#####################################################
### Define and fit the network health model
network_health_model = NetworkHealthModel(data)
network_health_model.fit(sample=2000, tune=1000)

### Get the model graph
filename = output_path + "/model_graph"
network_health_model.to_graphviz(filename)

### Save the model
rmtree(model_path, ignore_errors=True)
mkdir(model_path)

write_pickle_file(model_path, network_health_model, "network_health_model")


##########################################################
### Use the model prediction #############################
#########################################################
'''
### Load the network health model
network_health_model = read_pickle_file(model_path, "network_health_model")

## Compute relevant quantiles
percentiles=np.array([15, 25, 45, 85])
quantiles=network_health_model.get_quantiles(percentiles)
print(quantiles)


### Get the summary of a given parameter estimate
var_names=['alpha_health']
summary=network_health_model.get_summary(var_names=var_names)
print(summary)


### Get the summary of all parameters
sum=network_health_model.get_summary()
print(sum)

### Get network health stats for a group of people
filename=output_path + '/health_stats.csv'
person_ids=np.arange(0,500,1)
stats=network_health_model.get_network_health_stats(person_ids)
stats.to_csv(filename)


### Get stats of a given variable for a group of people
person_ids=[1,3,4]
var_names=['quality']
stats=network_health_model.get_stats(person_ids, var_names)


### Get network health status for a group of people
filename=output_path + '/health_proba.csv'
person_ids=np.arange(0,500,1)
status=network_health_model.get_network_health_proba(person_ids)
status.to_csv(filename)

### Get the posterior distribution of network health
var_names=["network_health"]
network_health_model.plot_posterior(var_names=var_names, figsize=(20,8), combine_dims={"observation"})
plt.savefig("outputs/network_health_distribution.png", dpi=150)
plt.show()
'''