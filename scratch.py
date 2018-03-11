import numpy as np
import matplotlib.pylab as plt
from utils import ADHD200, conform_1d, get_gc_config
from sklearn.model_selection import train_test_split
from gcforest.gcforest import GCForest
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets, input_data
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import progressbar
import warnings
import time
from pheno import get_model, predict
import pickle

warnings.filterwarnings("ignore")
t0 = time.time()


def check_and_get_pickled_data():
    try:
        with open('biomarkers.pkl', 'rb') as f:
            biomarkers = pickle.load(f)
        with open('adhd_labels.pkl', 'rb') as l:
            adhd_labels = pickle.load(l)
        print "Loaded Pickled Phenotypic info"
        return biomarkers, adhd_labels
    except IOError:
        return False, False


def generate_train_data(eq):
    print "Getting ADHD Train Data...\n"
    adhd_data = ADHD200()
    adhd_data.gen_data(eq=eq)
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
    print 'Data has {0} ADHD subjects.'.format(len(adhd_subjects)), '\n'
    bar.finish()
    return adhd_subjects, pooled_subjects, site_names, adhd_labels


def make_connectivity_biomarkers(kinds, labels, adhd200, func_files, pooled_subjects):
    new_labels = []
    connectivity_biomarkers = {}
    print 'error reloading (couldn\'t find pickle file... regenerating data'
    model = get_model(tts=0.2)
    for k in kinds:
        print "Measuring Connectivity for", k
        conn_measure = ConnectivityMeasure(kind=k, vectorize=True, discard_diagonal=True)
        connectivity = conn_measure.fit_transform(pooled_subjects)
        print 'Computing Phenotypes...'
        new_x = []
        bar = progressbar.ProgressBar(max_value=len(func_files))
        ops = 0
        for index in range(len(func_files)):
            probs = predict(model, adhd200, func_files[index])
            ops += 1
            bar.update(ops)
            if probs:
                new_labels.append(labels[index])
                features = np.array([conform_1d(probs, connectivity[index].shape[0]), connectivity[index]])
                new_x.append(features)
            else:
                print 'no phenotypic information found!'
                continue
        d3_dataset = np.array(new_x)
        nsamples, nx, ny = d3_dataset.shape
        d2_dataset = d3_dataset.reshape((nsamples, nx * ny))
        connectivity_biomarkers[k] = d2_dataset

    with open('biomarkers.pkl', 'wb') as bmks_file:
        pickle.dump(connectivity_biomarkers, bmks_file)

    with open('adhd_labels.pkl', 'wb') as lbls_file:
        pickle.dump(new_labels, lbls_file)

    return connectivity_biomarkers, new_labels


def run_model(kinds, biomarkers, adhd_labels, layers, f1=True, graph=False, report=True, c_matrix=False,
              print_results=False):
    accuracies = {}
    for k in kinds:
        #mlp = MLPClassifier(hidden_layer_sizes=(layers,), solver='lbfgs', verbose=0, random_state=0)
        # mlp = MLPClassifier(hidden_layer_sizes=(layers,), solver='lbfgs', verbose=0, random_state=0)
        classifier = GCForest(config=get_gc_config())
        #classifier = BaggingClassifier(base_estimator=mlp,  n_estimators=500, verbose=10)
        # classifier = SupervisedDBNClassification(hidden_layers_structure=[layers,])
        X_train, X_test, y_train, y_test = train_test_split(biomarkers[k], adhd_labels, test_size=0.2,
                                                            shuffle=True)
        classifier.fit_transform(np.array(X_train), np.array(y_train))
        print "Training with {0} training samples and {1} test samples".format(len(X_train), len(X_test))
        y_pred = classifier.predict(X_test)
        accuracies[k] = f1_score(y_test, y_pred)
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
        print max(accuracies, key=accuracies.get)
        print '\nF1 Scores'
        print accuracies
    if graph:
        return accuracies


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


def main(map, layers):
    t0 = time.time()
    kinds = ['tangent']
    pickled_bmks, pickled_lbls = check_and_get_pickled_data()
    if not pickled_bmks or not pickled_lbls:
        masker = get_atlas_data(map)

        adhd_data = generate_train_data(1.45)
        adhd_subjects, pooled_subjects, site_names, o_adhd_labels = get_adhd_dataset_info(adhd_data, masker)

        # kinds = ['correlation', 'partial correlation', 'tangent', 'covariance', 'precision']
        biomarkers, adhd_labels = make_connectivity_biomarkers(kinds, o_adhd_labels, adhd_data, adhd_data.func,
                                                               pooled_subjects)
    else:
        biomarkers, adhd_labels = pickled_bmks, pickled_lbls

    accuracies = run_model(kinds, biomarkers, adhd_labels, layers, graph=True,
                           c_matrix=True, print_results=True, f1=True)

    accuracies = accuracies.values()
    print "ran model", time.time() - t0, 'seconds'
    return accuracies


if __name__ == '__main__':
    main('sub-maxprob-thr50-2mm', 497)

