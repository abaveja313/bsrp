import numpy as np
import matplotlib.pylab as plt
from utils import ADHD200, TestADHD200
import time
from sklearn.model_selection import train_test_split
# from nolearn.dbn import DBN
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets, input_data
from sklearn.feature_selection import RFE
from nilearn.image import high_variance_confounds
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import progressbar
from sklearn.base import BaseEstimator
import warnings

warnings.filterwarnings("ignore")
t0 = time.time()


# TODO Combine train and test and use train_test_split
def generate_train_data(eq):
    print "Getting ADHD Train Data...\n"
    adhd_data = ADHD200()
    adhd_data.gen_data(eq=eq)
    return adhd_data


def generate_test_data(adhd200obj):
    print "Getting ADHD Test Data...\n"
    adhd200obj.gen_data(train=False)
    test_adhd200 = TestADHD200(adhd200obj)
    return test_adhd200


def get_msdl_data():
    print "Getting MSDL Atlas Data...\n"
    msdl_data = datasets.fetch_atlas_msdl()
    msdl_coords = msdl_data.region_coords
    n_regions = len(msdl_coords)
    print('MSDL has {0} ROIs, part of the following networks :\n{1}.'.format(
        n_regions, msdl_data.networks))
    masker = input_data.NiftiMapsMasker(
        msdl_data.maps, resampling_target="data", t_r=2.5, detrend=True,
        low_pass=.15, high_pass=.005, memory='nilearn_cache', memory_level=1, standardize=True)

    return msdl_data, msdl_coords, masker


def get_adhd_dataset_info(adhd_class, masker):
    print "Extracting info and masking NiFTi volumes...\n"
    adhd_subjects = []
    pooled_subjects = []
    site_names = adhd_class.site_names
    adhd_labels = adhd_class.labels  # 1 if ADHD, 0 if control
    print "\n"
    bar = progressbar.ProgressBar(max_value=len(adhd_class.func))
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
    print "\n"
    print('Data has {0} ADHD subjects.'.format(len(adhd_subjects)))
    bar.finish()
    return adhd_subjects, pooled_subjects, site_names, adhd_labels


def make_connectivity_biomarkers(kinds, pooled_subjects):
    connectivity_biomarkers = {}
    for k in kinds:
        print "measuring connectivity for", k
        conn_measure = ConnectivityMeasure(kind=k, vectorize=True, discard_diagonal=True)
        connectivity_biomarkers[k] = conn_measure.fit_transform(pooled_subjects)
    return connectivity_biomarkers


def print_report(grid_search, parameters):
    print()
    print("== " * 20)
    print("All parameters:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name, value in sorted(best_parameters.items()):
        if not isinstance(value, BaseEstimator):
            print("    %s=%r," % (param_name, value))

    print()
    print("== " * 20)
    print("Best score: %0.4f" % grid_search.best_score_)
    print("Best grid parameters:")
    for param_name in sorted(parameters.keys()):
        print("    %s=%r," % (param_name, best_parameters[param_name]))
    print("== " * 20)

    return grid_search


def run_model(kinds, biomarkers, adhd_labels, f1=True, graph=False, report=True, c_matrix=False,
              print_results=False):
    accuracies = []
    for k in kinds:
        clf = MLPClassifier(hidden_layer_sizes=(32,), learning_rate='adaptive', max_iter=300, random_state=0)
        rfe = RFE(SVC(kernel='linear', C=1.), 50, step=0.25)
        classifier = Pipeline([('rfe', rfe), ('mlp', clf)])
        # X_train, X_test, y_train, y_test = train_test_split(biomarkers[k], adhd_labels, test_size=0.25)
        scores = cross_val_score(classifier, biomarkers[k], adhd_labels, cv=10, scoring='f1')
        print "\n"
        print "= " * 20
        print "Biomarker:", k
        print "-" * 20
        print 'Scores:', scores
        print 'Mean Score:', np.mean(scores)
        print "Median:", np.median(scores)
        print "STD:", np.std(scores)
        print "= " * 20
        print '\n'
        # classifier.fit(X_train, y_train)
        # y_pred = classifier.predict(X_test)
        '''
        accuracies.append(f1_score(y_test, y_pred))
        if c_matrix:
            print '-' * 30
            print "Kind:", k
            print '\n', confusion_matrix(y_test, y_pred), '\n'

        if report:
            print "Classification Report:\n"
            print classification_report(y_test, y_pred, target_names=['No ADHD', 'ADHD'])
            print '\n'

        if print_results:
            print 'pred ' + str(list(y_pred))
            print 'true ' + str(list(y_test))
            print '\n\n'
        print "\nAccuracy:{0}".format(accuracy_score(y_test, y_pred))

    if f1:
        print "Best Connectome Strategy"
        print kinds[accuracies.index(max(accuracies))], '-', max(accuracies)
        print '\nF1 Scores'
        print accuracies
    if graph:
        return accuracies
'''


def draw_graph(kinds, accuracies):
    x = raw_input('\nWould you like to see a graph?\n')
    if x.lower() in ['y', 'yes']:
        plt.figure(figsize=(6, 4))
        positions = np.arange(len(kinds)) * .1 + .1
        plt.barh(positions, accuracies, align='center', height=.05)
        yticks = [kind.replace(' ', '\n') for kind in kinds]
        plt.yticks(positions, yticks)
        plt.xlabel('Max F1 Scores')
        plt.grid(True)
        plt.tight_layout()

        plt.show()
    else:
        quit()


def main():
    t0 = time.time()
    msdl_data, msdl_coords, masker = get_msdl_data()

    adhd_data = generate_train_data(1.45)
    adhd_subjects, pooled_subjects, site_names, adhd_labels = get_adhd_dataset_info(adhd_data, masker)

    kinds = ['correlation', 'partial correlation', 'tangent', 'precision']
    biomarkers = make_connectivity_biomarkers(kinds, pooled_subjects)

    accuracies = run_model(kinds, biomarkers, adhd_labels, graph=True,
                           c_matrix=True,
                           print_results=True, f1=True)

    print "ran model", time.time() - t0, 'seconds'

    if accuracies:
        draw_graph(kinds, accuracies)


if __name__ == "__main__":
    main()

'''
       classifier = DBN()

       parameters = {
           'layer_sizes': [
               [-1, 50, -1],
               [-1, 64, -1],
               [-1, 72, -1],
               [-1, int(round((743 + 2) ** 0.5)), -1]
           ],
           'epochs': [10, 50, 100, 500, 750, 1000],
           'learn_rates': [0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8],
           'scales': [0.01, 0.05, 0.1, 0.5],
           'dropouts': [0, 0.1, 0.25, 0.3],
           'epoc hs_pretrain': [0, 5, 7, 12, 20, 30],
           'learn_rate_decays': [0.1, 0.2, 0.7, 0.9, 1.3, 1.8],
           'verbose': [1]

       }

       grid_search = RandomizedSearchCV(estimator=classifier, n_iter=50, scoring='f1',
                                        param_distributions=parameters, verbose=10, cv=4)

       grid_search = grid_search.fit(biomarkers[k], adhd_labels)

       print_report(grid_search, parameters)
'''
