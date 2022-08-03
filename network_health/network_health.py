"""
This module define a NetworkHealthModel class for modeling and evaluation 
network health and other important indicators within an organization

"""
import os
import sys
from tempfile import TemporaryDirectory
import logging
import pickle
import shap
from termcolor import colored, cprint
import numpy as np
import pymc3 as pm
from sklearn.mixture import BayesianGaussianMixture
import arviz as az
import pandas as pd
from sklearn.model_selection import train_test_split
from hpsklearn import HyperoptEstimator
from hpsklearn import any_preprocessing
from hpsklearn import xgboost_regression
from hyperopt import tpe
from tqdm import tqdm
import hyperopt

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
#### /print debug information to stdout


class NetworkHealthModel:
    """
    Modelling network health
    """

    def __init__(self, data):
        self.model = generate_model(data)
        self.trace = None
        self.bgm_model = None
        self.coords = {"observation": data.index.values}
        self.stats = {}
        self.proba = {}
        self.classes = {}
        self.var_names = ["network_health"]

    def fit(self, sample: int = 2000, tune: int = 3000, target_accept: float = 0.95):
        """
        Fit the model using data
        """
        with self.model:
            self.trace = pm.sample(sample, tune=tune, target_accept=target_accept)

    def train_bgm_model(self, dist_stats: np.array, n_components: int, **kwargs):
        """
        Train a Bayasian Gaussian Mixtured model(bgm_model) to group network health distributions
        into clusters of similar distribution characteritics
        Return a trained bgm_model
        """
        self.bgm_model = BayesianGaussianMixture(n_components=n_components, **kwargs).fit(
            dist_stats
        )
        logging.info("The optimal classification is obtained")

    def get_classes(self, person_ids: list, classes: list, var_names: list = None, **kwargs):
        """
        Generate a model for classifying health distributions depending on their characteristics
        Classes are attributed to bgm model components depending on their mean (i.e, the component with
        the smallest mean will take the first class, whereas the component with the highest mean will take
        the last class)
        """
        if var_names is None:
            var_names = self.var_names

        n_components = len(classes)
        coords = {"observation": person_ids}
        stats = self.get_summary(var_names=var_names, kind="stats", coords=coords)
        stats_arr = stats.to_numpy()
        self.train_bgm_model(stats_arr, n_components, **kwargs)
        class_mean = dict(zip(list(range(n_components)), self.bgm_model.means_[:, 0]))
        class_mean = sort_dict(class_mean)
        self.classes={}
        for i, key in enumerate(class_mean.keys()):
            self.classes[key] = classes[i]

    def to_graphviz(self, filename: str, fileformat: str = "png"):
        """
        Export the model to a graph
        """
        graph = pm.model_to_graphviz(self.model)
        graph.unflatten(stagger=8)
        graph.render(filename=filename, format=fileformat)
        graph.view()

    def plot_trace(self, var_names: list, **kwargs):
        """
        Plot the distributions of a given set of variables
        """
        with self.model:
            az.plot_trace(self.trace, var_names=var_names, **kwargs)

    def get_summary(self, **kwargs):
        """
        Get the summary of relevant statistics
        """
        with self.model:
            summary = az.summary(self.trace, **kwargs)
        return summary

    def plot_posterior(self, var_names: list, **kwargs):
        """
        Plot the posterior distribution
        """
        with self.model:
            az.plot_posterior(self.trace, var_names, **kwargs)

    def extract_stats(self, person_ids: list):
        """
        Extract people network health statistics
        """
        stats = {id: self.stats.get(str(id), {}) for id in person_ids}
        return pd.DataFrame(list(stats.values()), index=list(stats.keys()))

    def extract_proba(self, person_ids: list):
        """
        Extract people network health probabilities
        """
        proba = {id: self.proba.get(str(id), {}) for id in person_ids}
        return pd.DataFrame(list(proba.values()), index=list(proba.keys()))

    def in_sample_predict(self, person_ids: list, var_names: list = None, **kwargs):
        """
        Make prediction with in sample data
        """
        if var_names is None:
            var_names = self.var_names

        stats = self.get_stats(person_ids, var_names, **kwargs)
        proba = self.get_proba(person_ids, stats, var_names)
        return stats, proba

    def get_stats(self, person_ids: list, var_names: list, **kwargs) -> pd.DataFrame:
        """
        Generate basic statistics of network health distributions
        """
        coords = {"observation": person_ids}
        summary = self.get_summary(var_names=var_names, kind="stats", coords=coords, **kwargs)
        people_stats = summary.to_dict(orient="index")
        people_stats = self.get_status(people_stats)
        if var_names[0] == self.var_names[0]:
            self.stats = self.save_records(person_ids, var_names[0], people_stats, self.stats)
        return pd.DataFrame(list(people_stats.values()), index=list(people_stats.keys()))

    def get_status(self, people_stats: dict) -> dict:
        """
        Determine people status using bgm model of classification
        """
        for people_id, stats in people_stats.items():
            people_data = np.array(list(stats.values())).reshape(1, -1)
            cluster = self.bgm_model.predict(people_data)
            stats["status"] = self.classes[cluster[0]]
            people_stats[people_id] = stats
        return people_stats

    def save_records(self, person_ids, var_name: str, data: dict, records: dict) -> dict:
        """
        Record data taking into account people ids
        """
        for person_id in person_ids:
            id_var = var_name + "[" + str(person_id) + "]"
            records[str(person_id)] = data[id_var]
        return records

    def get_proba(self, person_ids, stats: pd.DataFrame, var_names) -> pd.DataFrame:
        """
        Determine the probality of a given variable status
        """
        proba = {}
        data = stats.drop(columns=["status"])
        data_arr = data.to_numpy()
        proba = self.bgm_model.predict_proba(data_arr)
        proba_df = pd.DataFrame(index=list(stats.index), columns=list(self.classes.values()))
        for key, value in self.classes.items():
            proba_df[value]=proba[:,key]
        proba_df["status"] = stats["status"]
        proba_dict = proba_df.to_dict(orient="index")
        if var_names[0] == self.var_names[0]:
            self.proba = self.save_records(person_ids, var_names[0], proba_dict, self.proba)
        return proba_df

    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the data with x=log(1+exp(x))
        """
        data_adj = data.drop(columns=["people_strategic_align"])
        data_adj = data_adj.apply(lambda x: np.log(1 + np.exp(x)))
        data_adj["people_strategic_align"] = data["people_strategic_align"]
        return data_adj

    def predict(self, data: pd.DataFrame, var_names: list = None, **kwargs):
        """
        Predict distribution caracteristics and classification
        """
        if var_names is None:
            var_names = self.var_names

        data = normalize_data(data)
        coords = {"observation": data.index.values}
        with TemporaryDirectory(prefix="network_health") as tmpdir:
            write_pickle_file(tmpdir, self.model, "model")
            trace = self.trace
            with self.model:
                # Set up new data values
                pm.set_data(
                    {
                        "out_diffusion": data["out_diffusion"],
                        "in_diffusion": data["in_diffusion"],
                        "out_strategic_align": data["out_strategic_align"],
                        "in_strategic_align": data["in_strategic_align"],
                        "neighbor_out_holes": data["neighbor_out_holes"],
                        "neighbor_in_holes": data["neighbor_in_holes"],
                        "neighbor_out_clustering": data["neighbor_out_clustering"],
                        "neighbor_in_clustering": data["neighbor_in_clustering"],
                        "neighbor_out_cliques": data["neighbor_out_cliques"],
                        "neighbor_in_cliques": data["neighbor_in_cliques"],
                        "neighbor_out_betweeness": data["neighbor_out_betweeness"],
                        "neighbor_in_betweeness": data["neighbor_in_betweeness"],
                        "neighbor_out_closeness": data["neighbor_out_closeness"],
                        "neighbor_in_closeness": data["neighbor_in_closeness"],
                        "neighbor_out_degree": data["neighbor_out_degree"],
                        "neighbor_in_degree": data["neighbor_in_degree"],
                        "out_volume": data["out_volume"],
                        "in_volume": data["in_volume"],
                        "out_degree": data["out_degree"],
                        "in_degree": data["in_degree"],
                    }
                )
                posterior_predictive = pm.sample_posterior_predictive(
                    trace, var_names=var_names, keep_size=True, **kwargs
                )
                stats = self.get_predictive_stats(
                    coords["observation"], posterior_predictive, var_names
                )
                proba = self.get_proba(coords["observation"], stats, var_names)
            self.model = read_pickle_file(tmpdir, "model")
        return stats, proba

    def get_predictive_stats(self, person_ids, posterior_predictive, var_names: list, **kwargs):
        """
        Get summary statistics and network health status from predictions
        """
        summary = az.summary(posterior_predictive, var_names=var_names, kind="stats", **kwargs)
        summary = self.add_person_ids(summary, var_names, person_ids)
        people_stats = summary.to_dict(orient="index")
        people_stats = self.get_status(people_stats)
        if var_names[0] == self.var_names[0]:
            self.stats = self.save_records(person_ids, var_names[0], people_stats, self.stats)
        return pd.DataFrame(list(people_stats.values()), index=list(people_stats.keys()))

    def add_person_ids(self, summary: pd.DataFrame, var_names, person_ids) -> pd.DataFrame:
        """
        Add people ids to the sommary data
        """
        ids = [var_names[0] + "[" + str(i) + "]" for i in person_ids]
        summary["ids"] = ids
        return summary.set_index("ids")

    def predict2(self, data: pd.DataFrame, var_names: list = None, num_draws:int=None,**kwargs):
        """
        Make prediction using an adjusted trace
        """
        if var_names is None:
            var_names = self.var_names
        
        if num_draws is None:
            slice_draws=0
        else:
            slice_draws=-1*num_draws

        coords = {"observation": data.index.values}
        model = generate_model(data)
        variables = [
            "mean",
            "network_health",
            "neighbor_index",
            "volume_index",
            "quality",
            "diffusion",
            "efficiency",
            "strategic_align",
            "accessibility",
            "group_influence",
            "group_in_influence",
            "group_out_influence",
            "individual_influence",
            "individual_in_influence",
            "individual_out_influence",
        ]

        with TemporaryDirectory(prefix="network_health") as tmpdir:
            write_pickle_file(tmpdir, self.trace, "trace")
            trace= self.trace[slice_draws:]
            for var in variables:
                trace.remove_values(var)

            with model:
                posterior_predictive = pm.sample_posterior_predictive(
                    trace, var_names=var_names, keep_size=True, **kwargs
                )
                stats = self.get_predictive_stats(
                    coords["observation"], posterior_predictive, var_names
                )
                proba = self.get_proba(coords["observation"], stats, var_names)
            self.trace = read_pickle_file(tmpdir, "trace")
        return stats, proba
    
    def get_output(self, data: pd.DataFrame, output_names:list=None,  var_names: list = None, num_draws:int=None,**kwargs)->np.array:
        '''
        Get outputs of interest: ['mean'] or mean + class members (e.g. ['mean, 'poor', 'good'], etc.)
        '''
        stats, proba=self.predict2(data, var_names, num_draws, **kwargs)
        results=stats.join(proba, lsuffix='_1')
        results.drop(columns=['status_1'], inplace=True)
        if output_names is None:
            output_names=['mean'] + list(self.classes.values())
            return results[output_names].to_numpy()
        else:
            return results[output_names].to_numpy()
            

def generate_model(data: pd.DataFrame):
    '''Define a Bayesian Network model'''
    data = normalize_data(data)
    coords = {"observation": data.index.values}

    with pm.Model(coords=coords) as model:
        # read the data in
        people_strategic_align = pm.Data(
            "people_strategic_align", data["people_strategic_align"], dims="observation"
        )
        out_diffusion = pm.Data("out_diffusion", data["out_diffusion"], dims="observation")
        in_diffusion = pm.Data("in_diffusion", data["in_diffusion"], dims="observation")
        out_strategic_align = pm.Data(
            "out_strategic_align", data["out_strategic_align"], dims="observation"
        )
        in_strategic_align = pm.Data(
            "in_strategic_align", data["in_strategic_align"], dims="observation"
        )
        neighbor_out_holes = pm.Data(
            "neighbor_out_holes", data["neighbor_out_holes"], dims="observation"
        )
        neighbor_in_holes = pm.Data("neighbor_in_holes", data["neighbor_in_holes"], dims="observation")
        neighbor_out_clustering = pm.Data(
            "neighbor_out_clustering", data["neighbor_out_clustering"], dims="observation"
        )
        neighbor_in_clustering = pm.Data(
            "neighbor_in_clustering", data["neighbor_in_clustering"], dims="observation"
        )
        neighbor_out_cliques = pm.Data(
            "neighbor_out_cliques", data["neighbor_out_cliques"], dims="observation"
        )
        neighbor_in_cliques = pm.Data(
            "neighbor_in_cliques", data["neighbor_in_cliques"], dims="observation"
        )
        neighbor_out_betweeness = pm.Data(
            "neighbor_out_betweeness", data["neighbor_out_betweeness"], dims="observation"
        )
        neighbor_in_betweeness = pm.Data(
            "neighbor_in_betweeness", data["neighbor_in_betweeness"], dims="observation"
        )
        neighbor_out_closeness = pm.Data(
            "neighbor_out_closeness", data["neighbor_out_closeness"], dims="observation"
        )
        neighbor_in_closeness = pm.Data(
            "neighbor_in_closeness", data["neighbor_in_closeness"], dims="observation"
        )
        neighbor_out_degree = pm.Data(
            "neighbor_out_degree", data["neighbor_out_degree"], dims="observation"
        )
        neighbor_in_degree = pm.Data(
            "neighbor_in_degree", data["neighbor_in_degree"], dims="observation"
        )
        out_volume = pm.Data("out_volume", data["out_volume"], dims="observation")
        in_volume = pm.Data("in_volume", data["in_volume"], dims="observation")
        out_degree = pm.Data("out_degree", data["out_degree"], dims="observation")
        in_degree = pm.Data("in_degree", data["in_degree"], dims="observation")

        # Individual Influence
        # sigma=pm.Exponential("sigma", 1)
        gamma_inf = pm.HalfNormal("gamma_inf", 1)
        alpha_ind = (in_degree / (in_degree + out_degree), 1 - in_degree / (in_degree + out_degree))
        alpha_ind1 = pm.Dirichlet("alpha_ind1", a=np.ones(3))
        Sigma = pm.Exponential("Sigma", 1)

        individual_out_influence = pm.Deterministic(
            "individual_out_influence",
            (
                alpha_ind1[0] * (1 - np.exp(-(1 / gamma_inf) * neighbor_out_degree)) ** Sigma
                + alpha_ind1[1] * (1 - np.exp(-(1 / gamma_inf) * neighbor_out_closeness)) ** Sigma
                + alpha_ind1[2] * (1 - np.exp(-(1 / gamma_inf) * neighbor_out_betweeness)) ** Sigma
            )
            ** (1 / Sigma),
            dims="observation",
        )

        individual_in_influence = pm.Deterministic(
            "individual_in_influence",
            (
                alpha_ind1[0] * (1 - np.exp(-(1 / gamma_inf) * neighbor_in_degree)) ** Sigma
                + alpha_ind1[1] * (1 - np.exp(-(1 / gamma_inf) * neighbor_in_closeness)) ** Sigma
                + alpha_ind1[2] * (1 - np.exp(-(1 / gamma_inf) * neighbor_in_betweeness)) ** Sigma
            )
            ** (1 / Sigma),
            dims="observation",
        )

        individual_influence = pm.Deterministic(
            "individual_influence",
            (
                alpha_ind[0] * individual_in_influence ** Sigma
                + alpha_ind[1] * individual_out_influence ** Sigma
            )
            ** (1 / Sigma),
            dims="observation",
        )

        # Group Influence
        alpha_grp = (in_degree / (in_degree + out_degree), 1 - in_degree / (in_degree + out_degree))
        alpha_grp1 = pm.Dirichlet("alpha_grp1", a=np.ones(3))

        group_out_influence = pm.Deterministic(
            "group_out_influence",
            (
                alpha_grp1[0] * (1 - np.exp(-(1 / gamma_inf) * neighbor_out_cliques)) ** Sigma
                + alpha_grp1[1] * (1 - np.exp(-(1 / gamma_inf) * neighbor_out_clustering)) ** Sigma
                + alpha_grp1[2] * (1 - np.exp(-(1 / gamma_inf) * neighbor_out_holes)) ** Sigma
            )
            ** (1 / Sigma),
            dims="observation",
        )

        group_in_influence = pm.Deterministic(
            "group_in_influence",
            (
                alpha_grp1[0] * (1 - np.exp(-(1 / gamma_inf) * neighbor_in_cliques)) ** Sigma
                + alpha_grp1[1] * (1 - np.exp(-(1 / gamma_inf) * neighbor_in_clustering)) ** Sigma
                + alpha_grp1[2] * (1 - np.exp(-(1 / gamma_inf) * neighbor_in_holes)) ** Sigma
            )
            ** (1 / Sigma),
            dims="observation",
        )

        group_influence = pm.Deterministic(
            "group_influence",
            (alpha_grp[0] * group_in_influence ** Sigma + alpha_grp[1] * group_out_influence ** Sigma)
            ** (1 / Sigma),
            dims="observation",
        )

        # Accessibility
        alpha_ass = pm.Dirichlet("alpha_ass", a=np.ones(2))
        accessibility = pm.Deterministic(
            "accessibility",
            (alpha_ass[0] * individual_influence ** Sigma + alpha_ass[1] * group_influence ** Sigma)
            ** (1 / Sigma),
            dims="observation",
        )

        # Strategic Alignment
        alpha_strat = (in_volume / (in_volume + out_volume), 1 - in_volume / (in_volume + out_volume))
        strategic_align = pm.Deterministic(
            "strategic_align",
            (
                alpha_strat[0] * in_strategic_align ** Sigma
                + alpha_strat[1] * out_strategic_align ** Sigma
            )
            ** (1 / Sigma),
            dims="observation",
        )

        # Efficiency
        gamma_eff = pm.HalfNormal("gamma_eff", 1)
        efficiency = pm.Deterministic(
            "efficiency",
            np.exp((-1 / gamma_eff) * (in_strategic_align / out_strategic_align)),
            dims="observation",
        )

        # Diffusion
        alpha_dif = (in_volume / (in_volume + out_volume), 1 - in_volume / (in_volume + out_volume))
        diffusion = pm.Deterministic(
            "diffusion",
            (alpha_dif[0] * in_diffusion ** Sigma + alpha_dif[1] * out_diffusion ** Sigma)
            ** (1 / Sigma),
            dims="observation",
        )

        # Quality
        alpha_qual = pm.Dirichlet("alpha_qual", a=np.ones(3))
        quality = pm.Deterministic(
            "quality",
            (
                alpha_qual[0] * strategic_align ** Sigma
                + alpha_qual[1] * efficiency ** Sigma
                + alpha_qual[2] * diffusion ** Sigma
            )
            ** (1 / Sigma),
            dims="observation",
        )

        # Volume and Neighbor Index
        gamma_index = pm.HalfNormal("gamma_index", 1)
        volume_index = pm.Deterministic(
            "volume_index",
            1 - np.exp((-1 / gamma_index) * (in_volume + out_volume)),
            dims="observation",
        )

        neighbor_index = pm.Deterministic(
            "neighbor_index",
            1 - np.exp((-1 / gamma_index) * (in_degree + out_degree)),
            dims="observation",
        )

        # Network Health
        alpha_health = pm.Dirichlet("alpha_health", a=np.ones(2))
        network_health = pm.Deterministic(
            "network_health",
            (
                alpha_health[0] * (quality * volume_index) ** Sigma
                + alpha_health[1] * (accessibility * neighbor_index) ** Sigma
            )
            ** (1 / Sigma),
            dims="observation",
        )

        # People Strategic Alignment
        bias = pm.Normal("bias", mu=0, sd=1)
        weight = pm.HalfNormal("weight", 1)
        value = bias - weight * network_health
        mean = pm.Deterministic("mean", pm.math.sigmoid(value), dims="observation")
        std = pm.HalfNormal("std", sigma=1)

        pm.TruncatedNormal(
            "people_strategic_align_pred",
            mu=mean,
            sigma=std,
            observed=people_strategic_align,
            lower=0,
            upper=1,
            dims="observation",
        )
    return model


def normalize_data(data):
    """
    Normalize the data with x=log(1+exp(x))
    """
    data_adj = data.drop(columns=["people_strategic_align"])
    data_adj = data_adj.apply(lambda x: np.log(1 + np.exp(x)))
    data_adj["people_strategic_align"] = data["people_strategic_align"]
    return data_adj


def write_pickle_file(output_path: str, data, name: str):
    """
    Save data using pickle
    """
    filename = name + ".sav"
    if output_path is not None:
        pickle_path = os.path.join(output_path, filename)
    if os.path.isfile(pickle_path):
        os.remove(pickle_path)
    pickle.dump(data, open(pickle_path, "wb"))


def read_pickle_file(output_path: str, name: str):
    """
    Read data using pickle
    """
    filename = name + ".sav"
    if output_path is not None:
        pickle_path = os.path.join(output_path, filename)
    data = pickle.load(open(pickle_path, "rb"))
    return data


def sort_dict(var: dict) -> dict:
    """
    Sort a dictionary in ascending order
    """
    varlist = sorted((value, key) for (key, value) in var.items())
    sortvar = dict([(k, v) for v, k in varlist])
    return sortvar


class OutPutModel():
    '''
    A class to generate a model for a given output of network health
    '''
    def __init__(self, model, clone:bool=False, **kwargs) -> None:
        self.model=model
        self.output_names=kwargs.get('output_names',None)
        self.var_names=kwargs.get('var_names',None)
        self.num_draws=kwargs.get('num_draws',None)
        self.clone=clone

    def get_output(self, data:pd.DataFrame) -> np.array:
        '''
        Get an output based on inputs of clone or original model
        '''
        if self.clone:
            return self.model.get_output(data, output_names=self.output_names)
        else:
            return self.model.get_output(data, output_names=self.output_names,
                                                var_names=self.var_names,
                                                num_draws=self.num_draws)

class ShapExplainer():
    '''
    A class to compute shapley values
    '''
    def __init__(self, model, X_train:pd.DataFrame, clone:bool=False, **kwargs)->None:
        self.output_names=kwargs.get('output_names', None)
        self.var_names=kwargs.get('var_names', None)
        self.shap_values=None
        self.sample=None
        self.feature_names=kwargs.get('feature_names', None)
        self.output_model=OutPutModel(model, clone, **kwargs)
        self.explainer=shap.Explainer(self.output_model.get_output, X_train, output_names=self.output_names,feature_names=self.feature_names)

        if self.var_names is None:
            self.var_names=['network_health']

    def get_shap_values(self,sample:pd.DataFrame):
        '''
        Predict shapley values for a given sample
        '''
        shap_values=self.explainer(sample)
        self.shap_values=shap_values
        self.sample=sample
        return shap_values

    def display_local_shap_values(self, person_id:int, output_name:str, max_display=10)->None:
        '''
        Display shapley values for a given individual in the sample
        '''
        text=f"Features\'Impact on {self.var_names[0]} output ({output_name}): Person Id={person_id}"
        cprint(text, 'blue', attrs=['bold'])
        shap.plots.waterfall(self.shap_values[person_id,:, self.output_names.index(output_name)], max_display=max_display)


    def display_global_shap_values(self, output_name:str, max_display=10)->None:
        '''
        Display shapley values for a group of individuals
        '''
        text=f'Features\'Impact on {self.var_names[0]} output ({output_name}): Mean Absolute Impact'
        cprint(text, 'blue', attrs=['bold'])
        shap.plots.bar(self.shap_values[:,:,self.output_names.index(output_name)], max_display=max_display)
        text=f'Features\'Impact on {self.var_names[0]} output ({output_name}): Level Impact'
        cprint(text, 'blue', attrs=['bold'])
        shap.summary_plot(self.shap_values[:,:,self.output_names.index(output_name)], features=self.sample, 
                        feature_names=self.sample.columns, max_display=max_display)

    def get_clone_accuracy(self, clone_shap_values):
        '''
        Calculate the accuracy of shapley values between original and clone models
        '''
        clone_accuracy={}
        for i, output in enumerate(self.output_names):
            y_clone=clone_shap_values[:,:,i].values.flatten()
            y_orig=self.shap_values[:,:,i].values.flatten()
            e=y_orig-y_clone
            estimate=1-np.mean(np.power(e,2))/np.mean(np.power(y_orig,2))
            clone_accuracy[output]=estimate
        return pd.DataFrame([clone_accuracy])


class NetworkHealthClone():
    ''''
    A class to clone network health model
    '''
    def __init__(self, original_model:NetworkHealthModel, num_layers:int=1, clone_name1=xgboost_regression) -> None:
        self.original_model=original_model
        self.bgm_model= original_model.bgm_model
        self.classes=original_model.classes
        self.clone_model=[]
        self.clone_name1=clone_name1
        self.num_layers=num_layers
        self.stat_vars=['mean', 'sd','hdi_3%', 'hdi_97%']
        self.stats={}
        self.proba={}

    def fit(self, data, max_evals:int=30, n_repeats:int=10):
        '''
        Fit the clone model using data
        '''
        features=self.get_features(data)
        outputs=self.get_outputs(data)
        stat_vars=self.stat_vars
        clone_name1=self.clone_name1 
        self.clone_model=[]
        X=features

        for i in range(self.num_layers):
            layer=self.get_clone_layers(X, outputs, stat_vars, clone_name1, max_evals, n_repeats)
            output_hat=self.layer_predict(layer, stat_vars, X)
            X=np.concatenate((features, output_hat), axis=1)
            self.clone_model.append(layer)
    
    def get_optimal_regression_clone(self, X:np.ndarray, y:np.ndarray, clone_name,max_evals=30, n_repeats=10):
        '''
        Train a regression model to replicate the output of the network health model 
        '''
        clone=self.get_best_regression_clone(X, y, clone_name, max_evals, n_repeats)
        learn_clone=clone['learner']
        clone['learner']=learn_clone.fit(X,y)
        print("clone accuracy: %.3f" % clone['score'])
        return clone
     
    def get_best_regression_clone(self, X:np.ndarray, y:np.ndarray, clone_name,max_evals:int=30, n_repeats:int=10):
        '''
        Optimize the regression model based on data and hyperparameters
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        clones={}
        for i in tqdm(range(n_repeats), desc="Optimizing the best regression model on hyperparameters"):
            clone=self.get_regression_clone(X_train, y_train, clone_name, max_evals)
            score=clone.score(X_test, y_test)
            clones[str(i)]=(clone, score)
        keymax=max(clones, key=lambda x: clones[x][1])
        optimal_clone={'learner':clones[keymax][0], 'score':clones[keymax][1]}
        return optimal_clone

    def get_regression_clone(self, X:np.ndarray, y:np.ndarray, clone_name, max_evals:int=30):
        '''
        Identify the best regression model using hyperopt package
        '''
        clone= HyperoptEstimator(preprocessing=any_preprocessing('pre'),
                                regressor=clone_name('my_rf'),
                                algo=tpe.suggest,
                                trial_timeout=30,
                                max_evals=max_evals)
       
        clone.fit(X, y)     
        best_clone=clone.best_model()
        return best_clone['learner'].fit(X,y)

    def get_clone_layers(self, X:np.array, outputs:pd.DataFrame, stat_vars:list, clone_name1, max_evals:int=30, n_repeats:int=10):
        '''
        Train a layer of the clone network health model
        '''
        layer={}

        for var in stat_vars:
            y=outputs[var].to_numpy()
            clone_name=clone_name1
            clone=self.get_optimal_regression_clone(X, y, clone_name, max_evals, n_repeats)
            yhat=clone['learner'].predict(X)
            yhat=np.reshape(yhat, (-1,1))
            X=np.concatenate((X,yhat), axis=1)
            layer[var]=clone
        return layer

    def regularize_data(self, data):
        '''
        Regularize the data with the function log(1+exp(x))
        '''
        return data.apply(lambda x:np.log(1+np.exp(x)))

    def get_features(self, data:pd.DataFrame)->np.array:
        ''' Extract features from data and regularize'''
        if 'Unnamed: 0' in data.columns:
            X=data.drop(columns=['Unnamed: 0'])
        else:
            X=data
        X=self.regularize_data(X)
        return X.to_numpy()

    def get_outputs(self, data)->pd.DataFrame:
        ''' 
        Get network health model outputs
        '''
        stats, proba=self.original_model.predict2(data)
        outputs=stats.join(proba, lsuffix='_1')
        outputs.drop(columns=['status_1'], inplace=True)
        return outputs

    def layer_predict(self, layer:dict, stat_vars:list, X:np.array)->np.array:
        '''
        Make prediction using a clone model layer 
        '''
        num_x=X.shape[1]
        for var in stat_vars:
            yhat=layer[var]['learner'].predict(X)
            yhat=np.reshape(yhat, (-1,1))
            X=np.concatenate((X,yhat), axis=1)
        stats=X[:,num_x:]
        return stats
    
    
    def get_scores(self):
        '''
        Get the cloning accuracies
        '''
        scores={}
        for var, clone in self.clone_model[-1].items():
            scores[var]=clone['score']
        return scores
    
    def predict(self, data:pd.DataFrame)->pd.date_range:
        '''
        Make prediction using new data
        '''
        stats=self.get_stats(data)
        proba=self.get_proba(stats)
        return stats, proba

    def get_stats(self, data:pd.DataFrame)->pd.DataFrame:
        '''
        Predict network health stats
        '''
        stat_vars=self.stat_vars
        features=self.get_features(data)
        X=features
        for layer in self.clone_model:
            yhat=self.layer_predict(layer, stat_vars, X)
            X=np.concatenate((features, yhat), axis=1)
        stats=pd.DataFrame(yhat, index=data.index, columns=stat_vars)
        clusters = self.bgm_model.predict(yhat)
        status=self.convert_classes(clusters)
        stats['status']=status
        return stats

    def get_proba(self, stats:pd.DataFrame)->pd.DataFrame:
        '''
        Predict network health probabilities
        '''
        stats_arr=stats.drop(columns=['status']).to_numpy()
        proba_arr= self.bgm_model.predict_proba(stats_arr)
        proba=pd.DataFrame(index=stats.index, columns=list(self.classes.values()))
        for cluster, status in self.classes.items():
            proba[status]=proba_arr[:,cluster]
        proba['status']=stats['status']        
        return proba

    def convert_classes(self, clusters:np.array)->np.array:
        '''
        Convert clusters to classes
        '''
        results=[]
        for cluster in clusters.tolist():
            results.append(self.classes[cluster])
        return np.array(results)
    
    def get_output(self, data: pd.DataFrame, output_names:list=None)->np.array:
        '''
        Get an output of interest: 'mean' or a class members (e.g. 'poor', 'good', etc.)
        '''
        stats, proba=self.predict(data)
        results=stats.join(proba, lsuffix='_1')
        results.drop(columns=['status_1'], inplace=True)
        if output_names is None:
            output_names=['mean'] + list(self.classes.values())
            return results[output_names].to_numpy()
        else:
            return results[output_names].to_numpy()

def generate_data(num_obs:int=500)->pd.DataFrame:
    '''
    Generate synthetic data for network health model
    '''
    np.random.seed(1)
    rng = np.random.default_rng()
    N=num_obs
    in_degree=(1/N)*rng.integers(20, size=N)
    out_degree=(1/N)*rng.integers(30, size=N)
    in_volume=in_degree*rng.integers(40, size=N)
    out_volume=out_degree*rng.integers(50, size=N)
    neighbor_in_degree=(1/N)*rng.integers(20, size=N)
    neighbor_out_degree=(1/N)*rng.integers(30, size=N)
    neighbor_in_closeness=np.mean(rng.integers(2, size=(N-1,N)), axis=0)
    neighbor_out_closeness=np.mean(rng.integers(2, size=(N-1,N)), axis=0)
    neighbor_in_betweeness=(1/N)*rng.integers(30, size=N)
    neighbor_out_betweeness=(1/N)*rng.integers(40, size=N)
    neighbor_in_cliques=(1/N)*rng.integers(60, size=N)
    neighbor_out_cliques=(1/N)*rng.integers(30, size=N)
    neighbor_in_clustering=neighbor_in_degree*rng.beta(2,0.8,size=N)
    neighbor_out_clustering=rng.beta(2,0.8,size=N)*neighbor_out_degree*rng.beta(2,0.8,size=N)
    neighbor_in_holes=rng.beta(2,0.8,size=N)*(1-neighbor_in_clustering)
    neighbor_out_holes=rng.beta(2,0.8,size=N)*(1-neighbor_out_clustering)
    in_strategic_align=rng.beta(2,0.8,size=N)
    out_strategic_align=rng.beta(2,0.5,size=N)
    in_diffusion=rng.beta(3,1,size=N)
    out_diffusion=rng.beta(2,3,size=N)
    people_strategic_align=rng.beta(2,0.5,size=N)
    # Bundle data into datafrme
    data=pd.DataFrame({'people_strategic_align':people_strategic_align,
                        'out_diffusion': out_diffusion,
                        'in_diffusion':in_diffusion,
                        'out_strategic_align':out_strategic_align,
                        'in_strategic_align':in_strategic_align,
                        'neighbor_out_holes':neighbor_out_holes,
                        'neighbor_in_holes':neighbor_in_holes,
                        'neighbor_out_clustering':neighbor_out_clustering,
                        'neighbor_in_clustering':neighbor_in_clustering,
                        'neighbor_out_cliques':neighbor_out_cliques,
                        'neighbor_in_cliques':neighbor_in_cliques,
                        'neighbor_out_betweeness':neighbor_out_betweeness,
                        'neighbor_in_betweeness':neighbor_in_betweeness,
                        'neighbor_out_closeness':neighbor_out_closeness,
                        'neighbor_in_closeness':neighbor_in_closeness,
                        'neighbor_out_degree':neighbor_out_degree,
                        'neighbor_in_degree':neighbor_in_degree,
                        'out_volume':out_volume,
                        'in_volume':in_volume,
                        'out_degree':out_degree,
                        'in_degree':in_degree})
    return data