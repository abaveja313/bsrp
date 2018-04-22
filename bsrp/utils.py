# -*- coding:utf-8 -*-
import fnmatch
import os
import subprocess
import re
import pandas as pd
import numpy as np
import progressbar
from itertools import izip
import random
from pprint import pprint


class ADHD200(object):
    """
    This class is like the ADHD200 object included in the nilearn library. The ADHD200 object in nilearn, however,
    only contains 40 samples from the dataset, so this class was written to find files and phenotypic info, and
    structure it in a similar format to make it compatible with all of nilearn's APIs.
    Description of Attributes
    ----------------------
    - self.func: an array containing the functional MRI paths
    - self.ids: an array containg the ids of each MRI scan (used to retrieve phenotypic information)
    - self.root_dir: the path from which to locate files
    - self.train_glob: the glob selector to find fMRI scans, regardless of which number they are
    - self.id_regex: the regular expression to extract the id out of file names
    - self.pheno_regex: the regular expression to extract the phenotypic id out of the filename
    - self.pheno_filename: the phenotypic file containing the labels for each file
    - self.labels: the labels for each subject (0 for negative and 1 for positive)
    - self.site_names: the sites that correspond with each subject (KKI, NeuroIMAGE etc)
    """

    def __init__(self):
        self.func = []  # Paths of fMRI files
        self.ids = []  # The ids that correspond with each scan
        self.root_dir = '/Volumes/Amrit\'s SSD'  # For local
        # self.root_dir = '/home/ec2-user/data' # For AWS (Use Amazon Linux AMI)
        self.fmri_glob = 'sfnwmrda*_session_*_rest_1.nii'  # the glob file to select fMRI scans
        self.id_regex = '.*?\/([0-9].*?)\/'  # The RegEx to extract the id from the file path
        self.pheno_regex = '.*?\/train/(.*?)\/'  # the RegEx to extract the phenotypic information from the file path
        self.pheno_filename = 'phenotypic.csv'  # the global phenotypic file containing the labels
        self.labels = []  # the labels from each subject
        self.site_names = []  # the list of site names from each subject

    def _find_niis(self):
        """
        This function traverses through the root directory and appends functional file paths to the global func array.
        It also extracts the ids using the pheno regex and the site name and appends them to the global attributes
        """
        for root, dirnames, filenames in os.walk(
                self.root_dir + '/train'):  # Walk through all the files starting from the root directories
            for filename in fnmatch.filter(filenames,
                                           self.fmri_glob):  # Loop through all the files that match the fmri_glob
                path = os.path.join(root, filename)  # get the full path
                prefix = re.search(self.pheno_regex, path).group(
                    1)  # extract the pheno from the first match group using regex
                self.func.append(path)  # add it to the global func array
                self.ids.append(prefix + '_' + re.search(self.id_regex, path).group(1).lstrip(
                    '0'))  # extract the id and add it to the global id array
                self.site_names.append(prefix)  # add the site names

    def unzip_files(self, site_name):
        """
        This function goes through the specified site_name in from the root directory and unzips all of the
        files that match the fmri glob using gunzip
        :param site_name
        :return: None
        """
        for root, dirnames, filenames in os.walk(
                self.root_dir + '/train' + site_name):  # Walk through all the files from the site directory
            for filename in fnmatch.filter(filenames, self.fmri_glob + '.gz'):
                # extract all of the files that match the fmri_glob +
                file_name = os.path.join(root, filename)  # get the full path
                subprocess.call(['gunzip', '-k', file_name])  # unzip it using gunzip

    def _get_labels(self, pheno_filename, id):
        """
        Retrieve the phenotypic info for a given id
        :param pheno_filename: the phenotypic filename
        :param id: the id for which to retrieve
        :return: the label for the subject
        """
        df = pd.read_csv(self.root_dir + '/' + pheno_filename)  # Turn the phenotypic info into a pandas dataframe
        return df[df['Subject'] == id]['Group']  # Return the corresponding label for each subject

    def gen_data(self):
        """
        Generate the ADHD200 dataset data and skip the sample if there is no label
        """
        temp_func = []
        temp_sites = []
        temp_labels = []  # Initialize the temp files to []
        self._find_niis()  # Find the NiFTi files
        ops = 0  # initialize progressbar to 0
        bar = progressbar.ProgressBar(max_value=len(self.ids))
        for index, id in enumerate(self.ids):  # Loop through the array index and id for the global ids
            ops += 1  # add one to the progress bar
            bar.update(ops)  # update it to the current value of ops
            label = self._get_labels(self.pheno_filename, id)
            if label.empty:  # If there is no label for that scan
                continue  # Skip this sample
            else:
                temp_func.append(self.func[index])  # add it to the temp_func, so that we only keep samples for which
                # we have a label
                temp_sites.append(self.site_names[index])  # add the site to the temp_sites
                if 'ADHD' in label.tolist()[0]:  # if the subject is ADHD
                    temp_labels.append(1)  # append 1 to the temp_labels
                else:  # otherwise
                    temp_labels.append(0)  # append 0 to the temp_labels

        self.func = temp_func  # set the global func files to the temp_func
        self.site_names = temp_sites  # set the global site names to the temp_sites
        self.labels = temp_labels  # add the labels
        bar.finish()  # finish the bar

    def retrieve_pheno_for_model(self, index):
        """
        Retrieve the phenotypic information for either a given func path or an index in the functional array.
        It returns the age, dexterity and gender of each subject.
        :param index: either an fMRI path or an integer index from the func array
        :return: False if no information is found, or the pheno if it is found
        """
        if isinstance(index, int):  # if the index is an integer (the index)
            pheno = pd.read_csv(self.root_dir + '/train/train_pheno/' + self.site_names[index] + '/phenotypic_comb.csv')
            # Read the phenotypic file depending on the site --> combined phenotypic file from train and test
            scanDirId = self.ids[index].split('_')[1]  # Get the id for the label from the input
            subject = pheno[pheno['ScanDirID'] == int(scanDirId)]  # Return all the phenotypic information
        else:  # If the index is a functional file
            ind = self.func.index(index)  # Find the index from the functional file (if it is a string)
            pheno = pd.read_csv(self.root_dir + '/train/train_pheno/' + self.site_names[ind] + '/phenotypic_comb.csv')
            # Read the phenotypic file depending on the site --> combined phenotypic file from train and test
            scanDirId = self.ids[ind].split('_')[1]  # Get the id for the label from the input
            subject = pheno[pheno['ScanDirID'] == int(scanDirId)]  # Return all of the subject's phenotypic information
        phenotypic_fields = [subject['Age'].values[0], subject['Gender'].values[0], subject['Handedness'].values[0]]
        # Retrieve the age, gender and dexterity of the subject
        if np.isnan(phenotypic_fields).any():  # if any of the values are null
            return None  # Return false
        else:
            return phenotypic_fields  # Return the phenotypic fields retireved


class Helpers:  # Some helper methods for the main.py file
    @staticmethod  # So the Helpers class doesn't have to be instantiated to use its functions
    def merge_two_dicts(dictionary1, dictionary2):
        """
        Given two dicts, merge them into a new dict as a shallow copy.

        :param dictionary1: one of the dictionaries to merge
        :param dictionary2: the other dictionary to merge
        :returns dictionary_merged: A merged dictionary, containg the keys and values from both of the inputs
        """

        dictionary_merged = dictionary1.copy()  # Copy the first dictionary into a new variable
        dictionary_merged.update(dictionary2)  # Add the second dictionary to the first (merge them)
        return dictionary_merged  # Return the merged dictionary

    @staticmethod
    def get_params(adhd200, func_file):
        """
        Retrieve the phenotypic information for a given fMRI file. If there is no phenotypic information, then
        return nothing

        :param adhd200: An instance of the ADHD200 object
        :param func_file: The functional file you would like phenotypic information for
        :return: None if there is no phenotypic information, otherwise the phenotypic information
        """

        retrieved_pheno = adhd200.retrieve_pheno_for_model(
            func_file)  # Retrieve the phenotypic info for a functional file
        if retrieved_pheno is not None:  # If there is phenotypic information
            return retrieved_pheno  # Return it
        return None  # Otherwise, return none

    @staticmethod
    def conform_1d(vals, target):
        """
        Conform a 1D array into being the same length as a target by adding zeros around it

        :param vals: an array of the values you want to conform
        :param target: the target length for the vals array to be
        :return: a list with the vals array and a bunch of zeros to conform to the length
        """

        new_array = [0] * target  # Create an array full of target # of 0's
        for i in range(0, len(vals)):  # Loop through the values
            new_array[i] = vals[i]  # Set the ith index of new_array to the ith index of vals
        return np.array(new_array)  # Return a numpy array with the conformed vals

    @staticmethod
    def _generate_dict_metrics(metric_name, data, label):
        """
        Generate a dictionary containing statistics for a given metric

        :param metric_name: The name of the metric e.g. accuracy
        :param data: The array containing the metric values from CV run
        :param label: whether the metric is for 0 or for 1
        :return: a dict containing the min, max, std, and mean values for the metric
        """

        if label == -1:  # If the label doesn't have positive or negative fields
            metric_type = ''  # Set the type to an empty string
        elif label == 0:  # If the label is non ADHD
            metric_type = 'neg_'  # Set the metric to negative
        else:  # Otherwise (the metric is adhd)
            metric_type = 'pos_'  # set the metric to positive

        return {  # Return a dict containing the mean, min, max, and std value for the metricx
            '{0}mean_{1}'.format(metric_type, metric_name): np.mean(data),
            '{0}min_{1}'.format(metric_type, metric_name): min(data),
            '{0}max_{1}'.format(metric_type, metric_name): max(data),
            '{0}std_{1}'.format(metric_type, metric_name): np.std(data)
        }

    @staticmethod
    def generate_csv_data(layer_order, accuracies, f1s, precisions, recalls, matrix):
        """
        Generate a dictionary of the metric portion of CSV Data

        :param layer_order: a string containing the estimator sequence
        :param accuracies: a 1D array containing the accurieces from each run
        :param f1s: a 2D array containg both the + and the - f1 scores from each CV run
        :param precisions: 2D array containg both the + and the - precisions from each CV run
        :param recalls: 2D array containg both the + and the - recalls from each CV run
        :param matrix: an array containing the values of each field in the confusion matrix from each run
        :return: the metric portion of the CSV data
        """
        data = {'layer_order': layer_order}  # Initialize the CSV data
        names = ['accuracy', 'f1', 'precision', 'recall', 'true_negative', 'false_positive', 'false_negative',
                 'true_positive']  # The names of the metrics
        metrics = [accuracies, f1s, precisions, recalls, matrix[0], matrix[1], matrix[2], matrix[3]]
        # The data of the metrics
        for metric_data, name in izip(metrics, names):  # Loop through a generator containing the metric and the name
            if name == 'accuracy' or 'false' in name or 'true' in name:  # if the data doesn't have +/- fileds
                data = Helpers.merge_two_dicts(Helpers._generate_dict_metrics(name, metric_data, -1), data)
                # Merge the generated mean, min, max and std data with the whole array
            else:
                data = Helpers.merge_two_dicts(Helpers._generate_dict_metrics(name, metric_data[0], 0), data)
                # Merge the negative generated mean, min, max and std data with the whole array
                data = Helpers.merge_two_dicts(Helpers._generate_dict_metrics(name, metric_data[1], 1), data)
                # Merge the positive generated mean, min, max and std data with the whole array
        return data

    initial_structure = {  # An array containing the mapping of the classifiers
        0: 'mlp',
        1: 'logit1',
        2: 'xgb',
        3: 'logit2',
        4: 'svc',
        5: 'logit3',
        6: 'rf'
    }

    fieldnames = ['layer_order', 'mlp_layers', 'mlp_solver', 'logistic_regressions', 'svc_kernel', 'xgb_estimators',
                  'rf_estimators', 'early_stopping_iterations', 'positions', 'mean_accuracy', 'max_accuracy',
                  'min_accuracy', 'std_accuracy', 'max_false_negative', 'max_false_positive', 'max_true_negative',
                  'max_true_positive', 'mean_false_negative', 'mean_false_positive', 'mean_true_negative',
                  'mean_true_positive', 'min_false_negative', 'min_false_positive', 'min_true_negative',
                  'min_true_positive', 'neg_max_f1', 'neg_max_precision', 'neg_max_recall', 'neg_mean_f1',
                  'neg_mean_precision', 'neg_mean_recall', 'neg_min_f1', 'neg_min_precision', 'neg_min_recall',
                  'neg_std_f1', 'neg_std_precision', 'neg_std_recall', 'pos_max_f1', 'pos_max_precision',
                  'pos_max_recall', 'pos_mean_f1', 'pos_mean_precision', 'pos_mean_recall', 'pos_min_f1',
                  'pos_min_precision', 'pos_min_recall', 'pos_std_f1', 'pos_std_precision', 'pos_std_recall',
                  'std_false_negative', 'std_false_positive', 'std_true_negative', 'std_true_positive']

    # The fieldnames for the CSV file that we export

    reversed_initial_structure = {  # The reverse mapping of the classifiers
        'mlp': 0,
        'logit1': 1,
        'xgb': 2,
        'logit2': 3,
        'svc': 4,
        'logit3': 5,
        'rf': 6
    }

    @staticmethod
    def write_attributes(id, cv, times, probability, maxes, choices, consider_options, out='attributes.txt'):
        """
        Write the runner's attributes to a text file
        Note- "\n" is the escape sequence for a new line in most programming languages

        :param id: the optimizer id
        :param cv: the number of iterations to run each model
        :param times: The number of different params that we test
        :param probability: The probability that an estimator will be included
        :param maxes: A dictionary containing the maximum values for random generation
        :param choices: A dictionary containing some choice values for mlp and svc
        :param consider_options: whether or not to include each model
        :param out: the text file to write to
        :return:
        """
        with open(out, 'w') as attributes_file:
            # Write all of the data to a text file so we can easily get this runners data
            attributes_file.write('OPTIMIZER ID: ' + id + '\n\n')
            attributes_file.write('Cross Validation: ' + str(cv) + '\n')
            attributes_file.write('Times To Run: ' + str(times) + '\n')
            attributes_file.write('Estimator Probability: ' + str(probability) + '\n\n')
            attributes_file.write('Max Values:\n')
            pprint(maxes, attributes_file)
            attributes_file.write('\n\n ')
            attributes_file.write('Model Attribute Choices:' + '\n')
            pprint(choices, attributes_file)
            attributes_file.write('\n\n ')
            attributes_file.write('Models To Consider' + '\n')
            pprint(consider_options, attributes_file)

    @staticmethod
    def decision(probability):
        """
        A function that has a <probability> chance of returning True
        :param probability: float - the probability that this function returns true
        :return: True or False
        """
        return random.random() < probability  # Return the probability


def generate_gcforest_config(mlp_layers, mlp_solvers, number_logistic, svc_kernels, num_xgb_estims, rf_estims,
                             early_stopping_rounds, positions):
    """
    This function generates a configuration for the gcForest classifier from the random attributes specified
    by the Optimizer class

    :param mlp_layers: an array where each value is the number of nodes for mlp and the length is the number of hidden
        layers
    :param mlp_solvers: a list containing the MLP solvers for each MLPClassifier object
    :param number_logistic: a list containing the number of logistic regressions at each location
    :param svc_kernels: a list containing the SVC kernel for each SVC classifier in the cascade structure
    :param num_xgb_estims: a list containing the number of Gradient Boosting estimators for each XGB classifier
    :param rf_estims: a list containing the number of Random Forest estimators for each RF classifier
    :param early_stopping_rounds: the number of negative training iterations (where their accuracy is less then the
     previous) to abort training
    :param positions: The locations of each classifier in the sequence of estimators (see reversed_initial_structure)
    :return: the configuration
    """
    config = {}
    ca_config = {}  # Initialize the config
    ca_config["random_state"] = 0  # Set the random seed to 0
    ca_config["max_layers"] = 1000  # Set the gcForest's max layers to 1000
    ca_config["early_stopping_rounds"] = early_stopping_rounds  # set the early stopping rounds to the arg specified
    ca_config["n_classes"] = 2  # set the number of classes to 2 (only adhd and no adhd)
    estimators_to_append = []  # Intialize the list of estimator sequences
    keys = []  # Intialize the list of keys
    ca_config["estimators"] = [0] * len(positions)  # Create an array positions long

    if mlp_layers and mlp_solvers:  # If MLP was chosen to be included
        for mlp_layer, mlp_solver in zip(mlp_layers, mlp_solvers):  # Loop through each classifier
            estimators_to_append.append(
                {
                    "n_folds": 4,
                    "type": "MLPClassifier",
                    "hidden_layer_sizes": mlp_layer,
                    "random_state": 0,
                    'solver': mlp_solver
                }
            )  # add it to the array of classifiers
            keys.append(Helpers.reversed_initial_structure['mlp'])  # append its index to the keys array

    for _ in range(number_logistic[0]):  # Loop through the first logistic estimator number
        estimators_to_append.append({"n_folds": 4, "type": "LogisticRegression"})  # add a logistic classifier
        keys.append(Helpers.reversed_initial_structure['logit1'])  # add its index to the keys array

    if num_xgb_estims:  # If XGB was chosen to be included
        for xgb_estim in num_xgb_estims:  # Loop through each classifier
            estimators_to_append.append(
                {"n_folds": 5,
                 "type": "XGBClassifier",
                 "n_estimators": xgb_estim,
                 "objective": "multi:softprob",
                 "silent": True,
                 "nthread": -1,
                 "num_class": 2,

                 }
            )  # add it to the array of classifiers
            keys.append(Helpers.reversed_initial_structure['xgb'])  # append its index to the keys array

    for _ in range(number_logistic[1]):  # Loop through the second logistic estimator number
        estimators_to_append.append({"n_folds": 4, "type": "LogisticRegression"})  # add it to the array of classifiers
        keys.append(Helpers.reversed_initial_structure['logit2'])  # append its index to the keys array

    if svc_kernels:  # if svc was chosen to be included
        for kernel in svc_kernels:  # Loop through each classifier
            estimators_to_append.append(
                {
                    "n_folds": 5,
                    "type": "SVC",
                    "random_state": 0,
                    "kernel": kernel,
                    "probability": True
                }
            )  # add it to the array of classifiers
            keys.append(Helpers.reversed_initial_structure['svc'])  # append its index to the keys array

    estimators_to_append.append({"n_folds": 4, "type": "LogisticRegression"})  # add a logistic regression
    keys.append(Helpers.reversed_initial_structure['logit3'])  # add its index to the keys array

    if rf_estims:  # if we chose to include random forests
        for estim in rf_estims:  # loop through each classifier
            estimators_to_append.append(
                {"n_folds": 5,
                 "type": "RandomForestClassifier",
                 "n_estimators": estim,
                 "max_depth": None,
                 "n_jobs": -1,
                 }
            )  # add it to the array of classifiers
            keys.append(Helpers.reversed_initial_structure['rf'])  # add its index to the keys array
    if positions:  # if we chose to shuffle the positions arround
        for index, key in enumerate(keys):  # Loop through the index and keys of the keys array
            if key in positions:  # if the key exists in the positions array
                indices = [i for i, x in enumerate(positions) if
                           x == key]  # find all indexes where that classifer appears in the positions array
                for i in indices:  # loop through the indexes
                    ca_config['estimators'][i] = estimators_to_append[i]  # set the ith estimator of
                    #  estimators_to_append as the ith estimator of the config

    else:  # if we didn't choose to shuffle
        ca_config['estimators'] = estimators_to_append  # set the config to the original order

    # print 'Using config', ca_config  # for debugging

    config["cascade"] = ca_config  # set the config to the gcForest config
    return config  # return it
