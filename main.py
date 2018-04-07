import numpy as np
from utils import ADHD200, conform_1d, get_gc_config, fieldnames, merge_two_dicts
from sklearn.model_selection import train_test_split
from gcforest.gcforest import GCForest
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets, input_data
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from progressbar import ProgressBar
from time import time
from pheno import get_params
from pickle import load, dump
from pprint import pprint
from random import randint, choice, shuffle
from csv import DictWriter
from os.path import exists


def check_and_get_pickled_data():
    """
    Check and see if features/labels are previously cached via pickle.

    :return: false, false if no cached files are found
    :return: features, labels if they are
    """

    try:
        with open('features.pkl', 'rb') as featureFile:  # Try to retrieve the cached features (using pickle)
            features = load(featureFile)

        with open('adhd_labels.pkl', 'rb') as labelFile:  # Try to retrieve the cached labels (using pickle)
            adhd_labels = load(labelFile)

        print "Loaded Pickled Phenotypic info"
        return features, adhd_labels  # If they are both found, return them

    except IOError:  # If it can't find them and throws an error, return false, false as the features and labels
        # need to be recomputed
        return False, False


def generate_train_data():
    print "Getting ADHD Train Data...\n"
    adhd_data = ADHD200()
    adhd_data.gen_data()
    return adhd_data


def get_atlas_data(map):
    dataset = datasets.fetch_atlas_harvard_oxford(map)
    atlas_filename = dataset.maps

    masker = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                                          memory='nilearn_cache')

    return masker


def get_adhd_dataset_info(adhd_class, masker):
    print  "Extracting info and masking NiFTi volumes...\n"
    adhd_subjects = []
    pooled_subjects = []
    site_names = adhd_class.site_names
    adhd_labels = adhd_class.labels  # 1 if ADHD, 0 if control
    print "\n"
    bar = ProgressBar(max_value=len(adhd_class.func))
    ops = 0
    # print 'confounds', adhd_data.confounds
    for func_file, phenotypic, is_adhd, site in zip(
            adhd_class.func, adhd_class.pheno, adhd_labels, adhd_class.site_names):
        bar.update(ops)
        ops += 1
        # confounds = high_variance_confounds(func_file)
        # time_series = masker.fit_transform(func_file, confounds=confounds)
        time_series = masker.fit_transform(func_file)

        pooled_subjects.append(time_series)
        if is_adhd == 1:
            adhd_subjects.append(time_series)
        # adhd_labels.append(is_adhd)
    print 'Data has {0} ADHD subjects.'.format(len(adhd_subjects)), '\n'
    bar.finish()
    return adhd_subjects, pooled_subjects, site_names, adhd_labels


def make_connectivity_biomarkers(kinds, labels, adhd200, func_files, pooled_subjects):
    new_labels = []
    connectivity_biomarkers = {}
    print 'error reloading (couldn\'t find pickle file... regenerating data'
    for k in kinds:
        print "Measuring Connectivity for", k
        conn_measure = ConnectivityMeasure(kind=k, vectorize=True, discard_diagonal=True)
        connectivity = conn_measure.fit_transform(pooled_subjects)
        print 'Computing Phenotypes...'
        new_x = []
        bar = ProgressBar(max_value=len(func_files))
        ops = 0
        for index in range(len(func_files)):
            inp = get_params(adhd200, func_files[index])
            ops += 1
            bar.update(ops)
            if inp:
                new_labels.append(labels[index])
                features = np.array([conform_1d(inp, connectivity[index].shape[0]), connectivity[index]])
                # pprint(features)
                new_x.append(features)
            else:
                print 'no phenotypic information found!'
                continue
        d3_dataset = np.array(new_x)
        nsamples, nx, ny = d3_dataset.shape
        d2_dataset = d3_dataset.reshape((nsamples, nx * ny))
        connectivity_biomarkers[k] = d2_dataset

    with open('biomarkers.pkl', 'wb') as bmks_file:
        dump(connectivity_biomarkers, bmks_file)

    with open('adhd_labels.pkl', 'wb') as lbls_file:
        dump(new_labels, lbls_file)

    return connectivity_biomarkers, new_labels


def run_model(kind, features, adhd_labels, rand_params, verbose=True, test_size=0.2):
    """
    Run the gcForest using parameters from the Optimizer. Use random portions of the original dataset
    for testing and training (default 20%-80%)

    :param kind: The type of functional connectivity we want to use
    :param features: A matrix containing phenotypic and functional connectivity c
    :param adhd_labels: The correct labels from the dataset
    :param rand_params: The generated random params from the Optimizer
    :param verbose: Whether to print classification report
    :param test_size: How much of the dataset to use for testing
    :return:
    """
    classifier = GCForest(  # Instantiate the gcForest algorithm using the random parameters we generated
        config=get_gc_config(rand_params['mlp_layers'], rand_params['mlp_solver'], rand_params['logistic_regressions'],
                             rand_params['svc_kernel'], rand_params['xgb_estimators'],
                             rand_params['early_stopping_iterations'], rand_params['positions'])
    )

    X_train, X_test, y_train, y_test = train_test_split(features[kind], adhd_labels, test_size=test_size)
    # Split the data into random subsets (20% test, 80% train by default)
    classifier.fit_transform(np.array(X_train), np.array(y_train))  # Train the gcForest model
    y_pred = classifier.predict(X_test)  # Predict off of the test dataset

    if verbose:
        print classification_report(y_test, y_pred)  # Print out some useful run information

    scores = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(
        y_test, y_pred)  # Get the accuracy, f1, precision and recall of the model

    return scores  # Return it


class Optimizer(object):
    """
    This class randomly selects parameters (models and their attributes) for the gcForest algorithm. It only manipulates
    models that have shown promise, and only manipulates attributes that have an influence on results

    Models considered:

    - Multi Layer Perceptron (MLP)
    - SVC
    - Logistic Regression
    - Gradient Boosting (xgb)

    What we change randomly:

    - Whether to include each model
    - The number of hidden layers and nodes in the MLP (randomly selected up to 4 layers and 150 nodes
    - The number of logistic regressions between the SVC and XGB, and the XGB and MLP
    - The MLP Solver (adam, lbfgs, sgd)
    - The SVC Kernel (poly, rbf, linear)
    - The number of XGB estimators (up to 100)
    - The order of the models
    - The early stopping parameter (how many iterations in a row need to be below the best average for the model to be terminated)

    """

    def __init__(self):
        self.map = 'sub-maxprob-thr50-2mm'  # This is the brain atlas we use for normalization (from HarvardOxford)
        self.CV = 5  # The number of random subsets to test our data on
        self.connectivity_metric = 'tangent'  # The type of functional connectivity extraction we use
        self.times_to_run = 1000  # Number of times that we randomly generate and test params before ending
        self.verbose = True
        self.csv = True
        self.textFileOut = False
        pickled_features, pickled_labels = check_and_get_pickled_data()  # Check and see if biomarkers are already created
        if not pickled_features or not pickled_labels:  # If we don't already have the data cached locally
            masker = get_atlas_data(self.map)  # Generate a mask using the HarvardOxford atlas
            adhd_data = generate_train_data()  # Retrieve the data from my hard drive
            adhd_subjects, pooled_subjects, site_names, non_filtered_labels = get_adhd_dataset_info(adhd_data, masker)
            features, adhd_labels = make_connectivity_biomarkers(self.connectivity_metric, non_filtered_labels,
                                                                 adhd_data, adhd_data.func, pooled_subjects)
            # Calculate functional connectivity and combine phenotypic information as a feature. Returns a matrix
            # containing phenotypic information and computed functional connectivity -> features
        else:
            features, adhd_labels = pickled_features, pickled_labels  # If it is cached, retrieve it

        self.features = features
        self.labels = adhd_labels

    def _csv_writer(self, iteration_input, iteration_output, csvFile='results.csv'):
        """
        Write outputs to csv file

        :param iteration_input: the parameters that werre into the model
        :param iteration_output: the returned result from the CV
        :param csvFile: path to a csv file for outputing
        :return: None
        """
        if self.csv:
            is_new_file = not exists(csvFile)  # Check if the file exists so we know whether to write the header
            with open(csvFile, 'a') as csv_file:  # Open the csv file for appending
                writer = DictWriter(csv_file, fieldnames=fieldnames)  # Initialize a csv writer (using dictionary)
                if is_new_file:
                    writer.writeheader()  # If the file is new, create a header
                writer.writerow(merge_two_dicts(iteration_input, iteration_output))  # Write a new merged dictionary
                # of input and output (using the fieldset in utils.py)

    def _random_args(self, max_layers=3, max_regressions=4, max_nodes=150, max_xgb=100, max_early_stopping=5,
                     consider_mlp=True, consider_svc=True, consider_logit=True, consider_xgb=True,
                     consider_early_stopping=True, shuffle_models=True, verbose=True):
        """
        Generate random parameters for testing

        :param max_layers: maximum number of layers to consider for MLP
        :param max_regressions: maximum number of logistic regressions to consider
        :param max_nodes: maximum number of nodes to consider inside layers for MLP
        :param max_xgb: maximum number of xgb estimators to consider
        :param max_early_stopping: the maximum number of early stopping itertions to consider
        :param consider_mlp: whether or not to include multi layer perceptron in the optimization
        :param consider_svc: whether or not to include SVC in the optimization
        :param consider_logit: whether or not to include Logistic Regression in the optimization
        :param consider_xgb: whether or not to include gradient boosting in the optimization
        :param consider_early_stopping: whether or not to randomize the "down" iterations required to stop a model train
        :param shuffle_models: whether or not to shuffle the order of the models around
        :param verbose: whether to print out the config every time it is computed
        :return: dict: a dictionary contating the model parameters (MLP Solver, MLP Layers, XGB Estimators, number of
            Logistic Regressions and SVC kernel)
        """

        mlp = choice([True, False])  # Decide whether to include MLP
        svc = choice([True, False])  # Decide whether to include SVC
        xgb = choice([True, False])  # Decide whether to include XGB
        logit = choice([0, 1, 2, 3])  # Decide whether to include one logit, 2 logit, or none

        if mlp and consider_mlp:
            mlp_solver = choice(['adam', 'sgd', 'lbfgs'])  # Randomly choose an MLP algorithm
            mlp_layer_schema = []
            number_of_layers = randint(1, max_layers)  # Randomly generate the number of layers between 1 and 3

            for layer in range(number_of_layers):  # Loop through all of the layers
                nodes_in_layer = randint(1, max_nodes)  # Generate the random number of nodes in each layer (up to max)
                mlp_layer_schema.append(nodes_in_layer)  # Add it to the array containing the MLP layer schema

        else:
            mlp_layer_schema = None
            mlp_solver = None

        if svc and consider_svc:
            svc_kernel = choice(['linear', 'poly', 'rbf'])  # Randomly choose a kernel for SVC
        else:
            svc_kernel = None

        if xgb and consider_xgb:
            xgb_estimators = randint(1, max_xgb)  # Generate a random number of XGB estimators (0 < num estim < 100)
        else:
            xgb_estimators = None

        if logit == 1 and consider_logit:
            number_of_logit_regressions = randint(1, max_regressions), 0  # Generate one group of logistic
            # regressions but leave the other blank
        elif logit == 2 and consider_logit:
            number_of_logit_regressions = randint(1, max_regressions), randint(0, max_regressions)  # Generate
            # two groups of a random number of logistic regressions
        elif logit == 3 and consider_logit:
            number_of_logit_regressions = 0, randint(1, max_regressions)  # Generate one group of logistic
            # regressions but leave the other blank
        else:
            number_of_logit_regressions = 0, 0

        if consider_early_stopping:
            early_stopping_iterations = randint(1, max_early_stopping)  # Randomly select the number of layers
            # required to stop the model iteration
        else:
            early_stopping_iterations = 2

        if shuffle_models:
            positions = shuffle([range(0, 5)])  # Shuffle the indexes of estimators in the config
        else:
            positions = None

        final_parameters = {
            'mlp_layers': mlp_layer_schema,
            'mlp_solver': mlp_solver,
            'svc_kernel': svc_kernel,
            'xgb_estimators': xgb_estimators,
            'logistic_regressions': number_of_logit_regressions,
            'early_stopping_iterations': early_stopping_iterations,
            'positions': positions
        }

        return final_parameters  # Return the final dictionary as a result

    def run(self):

        """
        Run the model self.times_to_run times using random parameters. Export the results to a CSV file
        """

        for times in range(0, self.times_to_run):  # Loop through the number of times we have to run
            output = {  # Initialize an empty dictionary containing the results from each iteration
                'accuracies': [],
                'f1s': [],
                'precisions': [],
                'recalls': []
            }
            random_attributes = self._random_args()  # Generate random model parameters
            for cv_run in range(self.CV):
                accuracy, f1, precision, recall = run_model(self.connectivity_metric, self.features, self.labels,
                                                            random_attributes)  # Run the model
                output['accuracies'].append(accuracy)  # Add this iteration's accuracy to the time's store
                output['f1s'].append(f1)  # Add this iteration's f1 to the time's store
                output['precisions'].append(precision)  # Add this iteration's precision to the time's store
                output['recalls'].append(recall)  # Add this iteration's recall to the time's store
                print 'Ran {0} times (iteration {1})'.format(times, cv_run), random_attributes

            data = {'mean_accuracy': np.mean(output['accuracies']),  # Create the table for CSV Writer
                    'max_accuracy': max(output['accuracies']),
                    'min_accuracy': min(output['accuracies']),
                    'std_accuracy': np.std(output['accuracies']),
                    'mean_f1': np.mean(output['f1s']),
                    'max_f1': max(output['f1s']),
                    'min_f1': min(output['f1s']),
                    'std_f1': np.std(output['f1s']),
                    'mean_precision': np.mean(output['precisions']),
                    'max_precision': max(output['precisions']),
                    'min_precision': min(output['precisions']),
                    'std_precision': np.std(output['precisions']),
                    'mean_recall': np.mean(output['recalls']),
                    'max_recall': max(output['recalls']),
                    'min_recall': min(output['recalls']),
                    'std_recall': np.std(output['recalls']),
                    }

            self._csv_writer(random_attributes, data)  # Write the data to a file


if __name__ == '__main__':
    optimization = Optimizer()
    optimization.run()
