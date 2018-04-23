# -*- coding:utf-8 -*-
import numpy as np
from utils import ADHD200, generate_gcforest_config, Helpers
from sklearn.model_selection import train_test_split
from gcforest.gcforest import GCForest
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets, input_data
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, \
    confusion_matrix
from progressbar import ProgressBar
from pickle import load, dump
from random import randint, choice, sample
from csv import DictWriter
from os.path import exists
from pprint import pprint


def check_and_get_pickled_data():
    """
    Check and see if features/labels are previously cached via pickle.

    :return: (bool) false, (bool) false if no cached files are found
    :return: (list) features, (list) labels if they are
    """

    try:
        with open('pickles/features.pkl', 'rb') as featureFile:  # Try to retrieve the cached features (using pickle)
            features = load(featureFile)

        with open('pickles/adhd_labels.pkl', 'rb') as labelFile:  # Try to retrieve the cached labels (using pickle)
            adhd_labels = load(labelFile)

        print "Loaded Pickled Phenotypic info"
        return features, adhd_labels  # If they are both found, return them

    except IOError:  # If it can't find them and throws an error, return false, false as the features and labels
        # need to be recomputed
        return False, False


def generate_train_data():
    """
    This function instantiates a new ADHD200 object and generates the dataset. It loops through the directories
    on my hard drive to find the files. It also find corresponding phenotypic information

    :return: (ADHD200) an ADHD200 instance
    """

    adhd_data = ADHD200()  # Instantiate the ADHD200 object
    adhd_data.gen_data()  # Generate the data
    return adhd_data  # Return it


def get_atlas_data(map):
    """
    This function takes a map from the HarvardOxford atlas and turns it into a mask we can apply to the NiFTi volumes
    :param map: the map in the HarvardOxford atlas to use
    :return: (NiftiLabelsMasker) a masker object
    """

    atlas = datasets.fetch_atlas_harvard_oxford(map)  # Retrieve the map from nilearn's datasets module
    map = atlas.maps  # Retrieve the maps from the atlas object

    masker = input_data.NiftiLabelsMasker(labels_img=map, standardize=True,
                                          memory='nilearn_cache')  # Turn it into a mask that we can apply

    return masker  # Return it


def apply_masks(functional_files, masker):
    """
    Apply the mask to all the fMRI scans and return them

    :param functional_files: (list) an array containing the paths to the fMRI scans
    :param masker: (NiftiLabelsMasker) the nilearn masker object we apply to each fMRI file
    :return: (list) array containing the masked fMRI scans
    """

    pooled_subjects = []  # Initialize an array for storing the mask applied subjects
    bar = ProgressBar(max_value=len(functional_files))  # Setup a progressbar so we can see progress and ETA
    ops = 0  # Setup the progressbar's initial length

    for func_file in functional_files:  # Loop through the functional files
        bar.update(ops)  # Update the progressbar to the current value of "ops"
        ops += 1  # Increment the bar by one

        time_series = masker.fit_transform(func_file)  # apply the mask to the functional file

        pooled_subjects.append(time_series)  # add it to our array

    bar.finish()  # complete the bar
    return pooled_subjects  # return the subjects


def make_connectivity_biomarkers(kind, labels, adhd200, pooled_subjects):
    """
    This function takes the masked fMRI volumes and the corresponding phenotypic information (age, gender and dexterity)
    and turns them into a 2D array for doing ML classification. If there is no phenotypic information available,
    we exlude it from the dataset.

    :param kind: (str) The type of functional connnectity we extract
    :param labels: (list) The truth values for the ADHD200 dataset
    :param adhd200: (ADHD200) The ADHD200 object
    :param pooled_subjects: (list) The masked fMRI volumes
    :return: (list) features, (list) labels
    """

    new_labels = []  # Initialize a new list for containing the new labels (only labels for fMRI volumes that
    # have corresponding phenotypic information
    temp_features = []  # Initialize a new list for containing the new labels (only labels for fMRI volumes that
    # have corresponding phenotypic information

    conn_measure = ConnectivityMeasure(kind=kind, vectorize=True, discard_diagonal=True)  # Generate the functional
    # connectivity using the biomarker specified
    connectivity = conn_measure.fit_transform(pooled_subjects)  # Apply it to all of the masked fMRI scans

    bar = ProgressBar(max_value=len(adhd200.func))  # Instantiate a new progressbar
    ops = 0  # Set the default value of the bar to 0

    for index in range(len(adhd200.func)):
        phenotypic_information = Helpers.get_params(adhd200, adhd200.func[
            index])  # Retrieve the corresponding phenotypic information for each fMRI
        ops += 1  # Increment the bar by one
        bar.update(ops)  # Update the progressbar to the value of the variable "ops"
        if phenotypic_information is not None:  # If we found phenotypic information for that fMRI
            new_labels.append(labels[index])  # Add it to the "approved" labels list
            generated_features = np.array(
                [Helpers.conform_1d(phenotypic_information, connectivity[index].shape[0]), connectivity[index]])
            # Add the phenotypic information and the functional connectivity as a matrix. We have to
            # surround the phenotypic information by 0s to make it the same shape as the connectivity (conform 1d)
            temp_features.append(generated_features)  # add it to the temp features
        else:
            continue  # Skip that fMRI scan from the dataset

    d3_dataset = np.array(temp_features)  # Convert the 3D temp_features array to a numpy array
    nsamples, nx, ny = d3_dataset.shape  # Extract the dimensionality of the data
    d2_functional_connectivity = d3_dataset.reshape((nsamples, nx * ny))  # Convert it to 2 dimensions

    with open('pickles/features.pkl',
              'wb') as features_file:  # Cache the features so that we don't have to run this
        # function again
        dump(d2_functional_connectivity, features_file)  # Dump them to the pickle file

    with open('pickles/adhd_labels.pkl', 'wb') as labels_file:  # Cache the biomarkers so that we don't have to run this
        # function again
        dump(new_labels, labels_file)  # Dump them to the pickle file

    return d2_functional_connectivity, new_labels  # Return them


def run_model(features, adhd_labels, rand_params, verbose=True, test_size=0.2):
    """
    Run the gcForest using parameters from the Optimizer. Use random portions of the original dataset
    for testing and training (default 20%-80%)

    :param kind: (str) The type of functional connectivity we want to use
    :param features: (list) A matrix containing phenotypic and functional connectivity c
    :param adhd_labels: (list) The correct labels from the dataset
    :param rand_params: (dict) The generated random params from the Optimizer
    :param verbose: (bool) Whether to print classification report
    :param test_size: (float) How much of the dataset to use for testing
    :return: (float) accuracy, (float) f1, (float) precision, (float) recall
    """
    classifier = GCForest(  # Instantiate the gcForest algorithm using the random parameters we generated
        config=generate_gcforest_config(rand_params['mlp_layers'], rand_params['mlp_solver'],
                                        rand_params['logistic_regressions'],
                                        rand_params['svc_kernel'], rand_params['xgb_estimators'],
                                        rand_params['rf_estimators'],
                                        rand_params['early_stopping_iterations'], rand_params['positions']),
    )

    X_train, X_test, y_train, y_test = train_test_split(features, adhd_labels, test_size=test_size)
    # Split the data into random subsets (20% test, 80% train by default)
    classifier.fit_transform(np.array(X_train), np.array(y_train))  # Train the gcForest model
    y_pred = classifier.predict(np.array(X_test))  # Predict off of the test dataset
    y_test = np.array(y_test)
    if verbose:
        print "Classification Report\n", classification_report(y_test, y_pred)  # Print out some useful run information
        print "Accuracy:", accuracy_score(y_test, y_pred)
        print "Confusion Matrix\n", confusion_matrix(y_test, y_pred)
    positive_metrics = {
        'f1': f1_score(y_test, y_pred),  # Calculate the f1 for class "1"
        'precision': precision_score(y_test, y_pred),  # Calculate the precision for class "1"
        'recall': recall_score(y_test, y_pred),  # Calculate the recall for class "1"
    }
    negative_metrics = {
        'f1': f1_score(y_test, y_pred, pos_label=0),  # Calculate the f1 for class "0"
        'precision': precision_score(y_test, y_pred, pos_label=0),  # Calculate the precision for class "0"
        'recall': recall_score(y_test, y_pred, pos_label=0),  # Calculate the recall for class "0"
    }
    matrix = confusion_matrix(y_test, y_pred)
    confusion = {  # Return the attributes of the confusion matrix
        'true_negative': matrix[0][0],  # Predicted false and is false
        'false_positive': matrix[0][1],  # Predicted false and is true
        'false_negative': matrix[1][0],  # Predicted true and is false
        'true_positive': matrix[1][1]  # Predicted true and is true
    }
    scores = accuracy_score(y_test, y_pred), positive_metrics, negative_metrics, confusion
    # Get the accuracy, f1, precision and recall of the model

    return scores  # Return it


class Optimizer(object):
    """
    This class randomly selects parameters (models and their attributes) for the gcForest algorithm. It only manipulates
    models that have shown promise, and only manipulates attributes that have an influence on results

    Models considered
    --------------------------
    - Multi Layer Perceptron (MLP)
    - SVC
    - Logistic Regression
    - Gradient Boosting (xgb)

    What we change randomly
    --------------------------

    - Whether to include each model
    - The number of hidden layers and nodes in the MLP (randomly selected up to 4 layers and 150 nodes
    - The number of logistic regressions between the SVC and XGB, and the XGB and MLP
    - The MLP Solver (adam, lbfgs, sgd)
    - The SVC Kernel (poly, rbf, linear)
    - The number of XGB estimators (up to 130 by default)
    - The number of Random Forest estimators (up to 130 by default)
    - The order of the models
    - The early stopping parameter (how many iterations in a row need to be below the best average for the model to be terminated)

    """

    def __init__(self, optimizer_id):
        self.map = 'sub-maxprob-thr50-2mm'  # This is the brain atlas we use for normalization (from HarvardOxford)
        self.CV = 4  # The number of random subsets to test our data on
        self.connectivity_metric = 'tangent'  # The type of functional connectivity extraction we use
        self.times_to_run = 3000  # Number of times that we randomly generate and test params before ending
        self.verbose = True  # Whether to print out model results after each iteration
        self.csv = True  # Whether to output results to a csv file or not
        self.estimator_chance = 0.5  # The chance that an estimator will be included

        self.shuffle_models = True  # Whether to shuffle the classifier order
        self.maxes = {  # Set maximum values for the random parameter generator
            'estimators': 2,
            'mlp_layers': 3,  # Maximum MLP Layers
            'mlp_nodes': 150,  # Maximum nodes in each MLP layer
            'xgb_trees': 130,  # Maximum number of XGB trees
            'rf_tress': 130,  # Maximum number of RF tress
            'early_stopping': 3  # Maximum number of early stopping
        }

        self.models_consider = {  # Which models to consider during optimization
            'rf': True,  # Include Random Forests?
            'xgb': True,  # Include XGB?
            'mlp': True,  # Include MLP?
            'svc': True,  # Include SVC?
            'logit': True,  # Include Logit?
        }

        self.classifier_atr_choices = {  # Certain choice attributes for classifiers
            'mlp': ['sgd', 'lbfgs', 'adam'],  # MLP solver choices
            'svc': ['rbf', 'linear', 'poly']  # SVC Kernel Choices
        }

        self.csvFile = 'optimizer_' + optimizer_id + '_metrics' + '.csv'
        # Set the CSV file name to include some useful information

        pickled_features, pickled_labels = check_and_get_pickled_data()  # Check and see if biomarkers are already
        # created

        try:
            if not pickled_features or not pickled_labels:  # If we don't already have the data cached locally
                masker = get_atlas_data(self.map)  # Generate a mask using the HarvardOxford atlas
                adhd_data = generate_train_data()  # Retrieve the data from my hard drive
                masked_fmris = apply_masks(adhd_data.func, masker)
                features, adhd_labels = make_connectivity_biomarkers(self.connectivity_metric, adhd_data.labels,
                                                                     adhd_data, masked_fmris)
                # Calculate functional connectivity and combine phenotypic information as a feature. Returns a matrix
                # containing phenotypic information and computed functional connectivity -> features
            else:
                features, adhd_labels = pickled_features, pickled_labels  # If it is cached, retrieve it
        except ValueError:
            features, adhd_labels = pickled_features, pickled_labels  # If it is cached, retrieve it

        self.features = features
        self.labels = adhd_labels

        Helpers.write_attributes(optimizer_id, self.CV, self.times_to_run, self.estimator_chance, self.maxes,
                                 self.classifier_atr_choices, self.models_consider)

    def _csv_writer(self, iteration_input, iteration_output):
        """
        Write outputs to csv file

        :param iteration_input: (dict) the parameters that were into the model
        :param iteration_output: (dict)the returned result from the CV
        :param csvFile: (string) path to a csv file for outputting
        :return: None
        """
        if self.csv:
            is_new_file = not exists(self.csvFile)  # Check if the file exists so we know whether to write the header
            with open(self.csvFile, 'a') as csv_file:  # Open the tsv file for appending
                writer = DictWriter(csv_file,
                                    fieldnames=Helpers.fieldnames,
                                    delimiter=',')  # Initialize a tsv writer (using dictionary)
                if is_new_file:
                    writer.writeheader()  # If the file is new, create a header
                writer.writerow(
                    Helpers.merge_two_dicts(iteration_input, iteration_output))  # Write a new merged dictionary
                # of input and output (using the fieldset in utils.py)

    def _find_order(self, random_params, verbose=True):
        """
        Create a string with an easilly readable version of the model sequence

        :param random_params: (dict) The output from the parameter randomizer
        :param verbose: (bool) whether or not you want to have the output printed
        :return: (str) a string containing the model sequence (x --> y --> z)
        """

        positions = random_params['positions']  # Get the "positions" field from the dict random_params
        order = "=>".join([Helpers.initial_structure[i] for i in positions])  # Get the corresponding models
        # to the order index numbers
        if verbose:
            print order  # if the user wants, print the order
        return order

    def _random_args(self, consider_mlp=True, consider_svc=True, consider_logit=True, consider_xgb=True,
                     consider_rf=True, consider_early_stopping=True):
        """
        Generate random parameters for testing

        :param consider_mlp: (bool) whether or not to include multi layer perceptron in the optimization
        :param consider_svc: (bool) whether or not to include SVC in the optimization
        :param consider_logit: (bool) whether or not to include Logistic Regression in the optimization
        :param consider_xgb: (bool) whether or not to include gradient boosting in the optimization
        :param consider_rf: (bool) whether or not to include random forests in the optimization
        :param consider_early_stopping: (bool) whether or not to randomize the "down" iterations required to stop a model train
        :return: (dict) a dictionary contating the model parameters (MLP Solver, MLP Layers, XGB Estimators, number of
            Logistic Regressions and SVC kernel)
        """

        mlp = Helpers.decision(probability=self.estimator_chance)  # Decide whether to include MLP
        svc = Helpers.decision(probability=self.estimator_chance)  # Decide whether to include SVC
        xgb = Helpers.decision(probability=self.estimator_chance)  # Decide whether to include XGB
        rf = Helpers.decision(probability=self.estimator_chance)  # Decide whether to include RF
        logit = choice([0, 1, 2, 3])  # Decide whether to include one logit, 2 logit, or none
        active_classifiers = []  # Keep a count so we know how many active estimators we have

        if mlp and self.models_consider['svc']:
            num_mlp = randint(0, self.maxes[
                'estimators'])  # How many MLP classifiers we should use in this round of testing
            mlp_layer_schema = []
            mlp_solvers = []
            for _ in range(num_mlp):  # Loop through all of the MLP classifiers we chose to consider
                temp_schema = []
                mlp_solvers.append(choice(self.classifier_atr_choices['mlp']))  # Randomly choose an MLP algorithm
                number_of_layers = randint(1, self.maxes['mlp_layers'])
                # Randomly generate the number of layers between 1 and 3
                active_classifiers.append(
                    Helpers.reversed_initial_structure['mlp'])  # Append so we keep track of active estimators
                for layer in range(number_of_layers):  # Loop through all of the layers
                    nodes_in_layer = randint(1, self.maxes['mlp_nodes'])
                    # Generate the random number of nodes in each layer (up to max)
                    temp_schema.append(nodes_in_layer)  # Add it to the array containing the MLP layer schema
                mlp_layer_schema.append(temp_schema)

        else:
            mlp_layer_schema = None
            mlp_solvers = None

        if svc and self.models_consider['svc']:
            num_svc = randint(0, self.maxes[
                'estimators'])  # Randomly generate the number of SVC classifiers we want to use
            svc_kernels = []
            for _ in range(num_svc):  # Loop through all of the SVC estimators we chose to consider
                svc_kernels.append(choice(self.classifier_atr_choices['svc']))  # Randomly choose a kernel for SVC
                active_classifiers.append(
                    Helpers.reversed_initial_structure['svc'])  # Append so we keep track of active estimators
        else:
            svc_kernels = None

        if xgb and self.models_consider['xgb']:
            num_xgb = randint(0, self.maxes['estimators'])
            xgb_estimators = []
            for _ in range(num_xgb):
                xgb_estimators.append(
                    randint(1, self.maxes['xgb_trees']))  # Generate a random number of XGB estimators (0 < num estim < 130)
                active_classifiers.append(
                    Helpers.reversed_initial_structure['xgb'])  # Append so we keep track of active estimators
        else:
            xgb_estimators = None

        if self.models_consider['logit']:
            if logit == 1:
                number_of_logit_regressions = randint(1, self.maxes['estimators']), 0  # Generate one group of logistic
                # regressions but leave the other blank
                for _ in range(number_of_logit_regressions[0]):
                    active_classifiers.append(
                        Helpers.reversed_initial_structure['logit1'])  # Append so we keep track of active estimators

            elif logit == 2:
                number_of_logit_regressions = randint(1, self.maxes['estimators']), randint(0, self.maxes['estimators'])
                # Generate two groups of a random number of logistic regressions
                for _ in range(number_of_logit_regressions[0]):
                    active_classifiers.append(
                        Helpers.reversed_initial_structure['logit1'])  # Append so we keep track of active estimators
                for _ in range(number_of_logit_regressions[1]):
                    active_classifiers.append(
                        Helpers.reversed_initial_structure['logit2'])  # Append so we keep track of active estimators

            elif logit == 3:
                number_of_logit_regressions = 0, randint(1, self.maxes['estimators'])  # Generate one group of logistic
                for _ in range(number_of_logit_regressions[1]):
                    active_classifiers.append(
                        Helpers.reversed_initial_structure['logit2'])  # Append so we keep track of active estimators
                # regressions but leave the other blank

            else:
                number_of_logit_regressions = 0, 0

        active_classifiers.append(
            Helpers.reversed_initial_structure[
                'logit3'])  # Add the constant logistic regressions to the active classifiers list

        if rf and consider_rf:
            number_of_rfs = randint(0, self.maxes['estimators'])  # Generate the number of RFs to consider
            rf_estims = []
            for _ in range(number_of_rfs):  # For each RF classifier
                active_classifiers.append(
                    Helpers.reversed_initial_structure['rf'])  # Add it to the active classifiers list
                rf_estims.append(randint(1, self.maxes['rf_tress']))  # Add the random number of estimators
        else:
            rf_estims = None

        if consider_early_stopping:
            early_stopping_iterations = randint(1, self.maxes['early_stopping'])  # Randomly select the number of layers
            # required to stop the model iteration
        else:
            early_stopping_iterations = 2

        if self.shuffle_models:
            positions = sample(active_classifiers, len(active_classifiers))
            # Shuffle the indexes of estimators in the config
        else:
            positions = active_classifiers

        final_parameters = {
            'mlp_layers': mlp_layer_schema,
            'mlp_solver': mlp_solvers,
            'svc_kernel': svc_kernels,
            'xgb_estimators': xgb_estimators,
            'logistic_regressions': number_of_logit_regressions,
            'rf_estimators': rf_estims,
            'early_stopping_iterations': early_stopping_iterations,
            'positions': positions,
        }
        print final_parameters
        return final_parameters  # Return the final dictionary as a result

    def run(self):

        """
        Run the model self.times_to_run times using random parameters. Export the results to a CSV file
        """

        for times in range(0, self.times_to_run):  # Loop through the number of times we have to run
            output = {  # Initialize an empty dictionary containing the results from each iteration
                'accuracies': [],
                'f1s_positive': [],
                'precisions_positive': [],
                'recalls_positive': [],
                'f1s_negative': [],
                'precisions_negative': [],
                'recalls_negative': [],
                'true_negative': [],
                'false_positive': [],
                'false_negative': [],
                'true_positive': []
            }

            random_attributes = self._random_args()  # Generate random model parameters
            layer_order = self._find_order(random_attributes)

            for cv_run in range(self.CV):
                accuracy, positive_metrics, negative_metrics, confusion_metrics = run_model(
                    self.features, self.labels,
                    random_attributes, verbose=True
                )
                #  Run the model and get metrics from that run
                output['accuracies'].append(accuracy)  # Add this iteration's metrics to the CV array
                output['f1s_positive'].append(positive_metrics['f1'])
                output['f1s_negative'].append(negative_metrics['f1'])
                output['precisions_positive'].append(positive_metrics['precision'])
                output['precisions_negative'].append(negative_metrics['precision'])
                output['recalls_negative'].append(negative_metrics['recall'])
                output['recalls_positive'].append(positive_metrics['recall'])
                output['true_negative'].append(confusion_metrics['true_negative'])
                output['false_positive'].append(confusion_metrics['false_positive'])
                output['false_negative'].append(confusion_metrics['false_negative'])
                output['true_positive'].append(confusion_metrics['true_positive'])

                print 'Ran {0} times (iteration {1})'.format(times, cv_run), random_attributes

            data = Helpers.generate_csv_data(layer_order, output['accuracies'],
                                             [output['f1s_negative'], output['f1s_positive']],
                                             [output['precisions_negative'], output['precisions_positive']],
                                             [output['recalls_negative'], output['recalls_positive']],
                                             [output['true_negative'], output['false_positive'],
                                              output['false_negative'], output['true_positive']])
            # Generate the dictionary for the CSV File

            self._csv_writer(random_attributes, data)  # Write the data to a file


if __name__ == '__main__':
    optimization = Optimizer(optimizer_id='001')
    optimization.run()
