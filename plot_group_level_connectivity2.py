import numpy as np
import matplotlib.pylab as plt
from make_dataObj import ADHD200, TestADHD200
import time
# from sklearn.svm import LinearSVC
# from dbn.tensorflow import SupervisedDBNClassification
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets, input_data
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import progressbar

t0 = time.time()


def generate_train_data(eq):
    print "Getting ADHD Train Data..."
    adhd_data = ADHD200()
    adhd_data.gen_data(eq=eq)
    return adhd_data


def generate_test_data(adhd200obj):
    print "Getting ADHD Test Data..."
    adhd200obj.gen_data(train=False)
    test_adhd200 = TestADHD200(adhd200obj)
    return test_adhd200


def get_msdl_data():
    print "Getting MSDL Atlas Data..."
    msdl_data = datasets.fetch_atlas_msdl()
    msdl_coords = msdl_data.region_coords
    n_regions = len(msdl_coords)
    print('MSDL has {0} ROIs, part of the following networks :\n{1}.'.format(
        n_regions, msdl_data.networks))
    masker = input_data.NiftiMapsMasker(
        msdl_data.maps, resampling_target="data", t_r=2.5, detrend=True,
        low_pass=.1, high_pass=.01, memory='nilearn_cache', memory_level=1, standardize=True)

    return msdl_data, msdl_coords, masker


def get_adhd_dataset_info(adhd_class, masker):
    print "Extracting info and masking NiFTi volumes..."
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
        conn_measure = ConnectivityMeasure(kind=k, vectorize=True)
        connectivity_biomarkers[k] = conn_measure.fit_transform(pooled_subjects)

    # For each kind, all individual coefficients are stacked in a unique 2D matrix.
    print('{0} correlation biomarkers for each subject.'.format(
        connectivity_biomarkers['correlation'].shape[1]))
    return connectivity_biomarkers


def run_model(kinds, biomarkers, preds, adhd_labels, y_true, f1=True, graph=False, report=True, c_matrix=False,
              print_results=False):
    accuracies = []
    for k in kinds:
        classifier = MLPClassifier()
        parameters = {'batch_size': [16, 32],
                      'hidden_layer_sizes': [(100, 50), (2, 2), (2, 2, 2), (256, 256)],
                      'learning_rate': ['constant', 'adaptive'],
                      'solver': ['sgd', 'adam'],
                      'shuffle': [True],
                      'random_state': [0],
                      'activation': ['relu', 'logistic'],
                      'verbose': [100]
                      }

        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=parameters,
                                   scoring='f1',
                                   verbose=100,
                                   cv=10)
        grid_search = grid_search.fit(biomarkers[k], adhd_labels)

        best_parameters = grid_search.best_params_
        best_accuracy = grid_search.best_score_
        print "best accuracy", best_accuracy
        print "best params", best_parameters

        # clf = LinearSVC(random_state=0)
        # y_pred = clf.predict(preds[k])
        # clf.save('model.pkl')
        '''
        accuracies.append(f1_score(y_true, y_pred))

        if c_matrix:
            print '-' * 30
            print "Kind:", k
            print '\n', confusion_matrix(y_true, y_pred), '\n'

        if report:
            print "Classification Report:\n"
            print classification_report(y_true, y_pred, target_names=['No ADHD', 'ADHD'])
            print '\n'

        if print_results:
            print 'pred ' + str(list(y_pred))
            print 'true ' + str(list(y_true))
            print '\n\n'

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
    test_adhd_data = generate_test_data(adhd_data)
    test_adhd_subjects, test_pooled_subjects, test_site_names, test_labels = get_adhd_dataset_info(test_adhd_data,
                                                                                                   masker)
    adhd_subjects, pooled_subjects, site_names, adhd_labels = get_adhd_dataset_info(adhd_data, masker)

    kinds = ['correlation', 'partial correlation', 'tangent', 'covariance', 'precision']

    biomarkers = make_connectivity_biomarkers(kinds, pooled_subjects)
    test_biomarkers = make_connectivity_biomarkers(kinds, test_pooled_subjects)

    accuracies = run_model(kinds, biomarkers, test_biomarkers, adhd_labels, test_labels, graph=True, c_matrix=True,
                           print_results=True, f1=True)

    print "ran model", time.time() - t0, 'seconds'

    if accuracies:
        draw_graph(kinds, accuracies)


if __name__ == "__main__":
    main()
