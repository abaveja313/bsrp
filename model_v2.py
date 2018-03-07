import matplotlib.pylab as plt
from utils import ADHD200, TestADHD200
import time
import numpy as np
from nolearn.dbn import DBN
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets, input_data
from sklearn.tree import DecisionTreeClassifier
from nilearn.image import high_variance_confounds
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import progressbar
import warnings

warnings.filterwarnings("ignore")

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
        low_pass=.15, high_pass=.005, memory='nilearn_cache', memory_level=1, standardize=True)

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

    # For each kind, all individual coefficients are stacked in a unique 2D matrix.
    print('{0} correlation biomarkers for each subject.'.format(
        connectivity_biomarkers['correlation'].shape[1]))
    return connectivity_biomarkers


def create_pheno_classifier(adhd200):
    X, y = adhd200.gen_pheno()
    X_test, y_test = adhd200.gen_pheno(train=False)
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    print 'Phenotypic Model'
    print classification_report(y_test, y_pred)
    return clf


def pheno_probability(clf, adhd200, nii_pred, weights=(2, 1)):
    y_pred = []
    for index, predicted_value in enumerate(nii_pred):
        res = adhd200.retrieve_pheno_for_model(index)
        if res:
            pheno_prediction = clf.predict_proba(res)
            #probability = np.average([predicted_value, pheno_prediction], weights=weights)
            y_pred.append(pheno_prediction)
            #y_pred.append(int(round(probability)))
        else:
            y_pred.append(predicted_value)
    return y_pred


def run_model(kinds, biomarkers, preds, adhd_labels, y_true, adhd200obj, clf, f1=True, graph=False, report=True,
              c_matrix=False,
              print_results=False, weights=(2, 1)):
    accuracies = []
    for k in kinds:
        classifier = DBN()
        classifier.fit(biomarkers[k], adhd_labels)
        nii_model_ouput = classifier.predict_proba(preds[k])
        print 'image model'
        print nii_model_ouput
        print y_true
        print '\n'
        y_pred = pheno_probability(clf, adhd200obj, nii_model_ouput)
        print y_pred
        print y_true

        '''
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
        print "\nAccuracy:{0}".format(accuracy_score(y_true, y_pred))
        accuracies.append(accuracy_score(y_true, y_pred))
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

    adhd_data = generate_train_data(1)
    test_adhd_data = generate_test_data(adhd_data)
    test_adhd_subjects, test_pooled_subjects, test_site_names, test_labels = get_adhd_dataset_info(test_adhd_data,
                                                                                                   masker)
    adhd_subjects, pooled_subjects, site_names, adhd_labels = get_adhd_dataset_info(adhd_data, masker)

    kinds = ['correlation', 'partial correlation', 'tangent', 'precision']

    biomarkers = make_connectivity_biomarkers(kinds, pooled_subjects)
    test_biomarkers = make_connectivity_biomarkers(kinds, test_pooled_subjects)
    clf = create_pheno_classifier(adhd_data)
    accuracies = run_model(kinds, biomarkers, test_biomarkers, adhd_labels, test_labels, adhd_data, clf, graph=True,
                           c_matrix=True, print_results=True, f1=True)

    print "ran model", time.time() - t0, 'seconds'

    if accuracies:
        draw_graph(kinds, accuracies)


if __name__ == "__main__":
    main()
