import numpy as np
import matplotlib.pylab as plt
from utils import ADHD200, conform_1d
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets, input_data
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import progressbar
import warnings
import time
from pheno import get_model, predict

warnings.filterwarnings("ignore")
t0 = time.time()


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
    model = get_model(tts=0.2)
    new_labels = []
    connectivity_biomarkers = {}
    for k in kinds:
        print "Measuring Connectivity for", k
        conn_measure = ConnectivityMeasure(kind=k, vectorize=True, discard_diagonal=True)
        connectivity = conn_measure.fit_transform(pooled_subjects)
        new_x = []
        for index in range(len(func_files)):
            probs = predict(model, adhd200, func_files[index])
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
    return connectivity_biomarkers, new_labels


def run_model(kinds, biomarkers, adhd_labels, layers, f1=True, graph=False, report=True, c_matrix=False,
              print_results=False):
    accuracies = {}
    for k in kinds:
        classifier = MLPClassifier(hidden_layer_sizes=(layers,), solver='lbfgs', verbose=0, random_state=0)
        # classifier = SupervisedDBNClassification(hidden_layers_structure=[layers,])
        X_train, X_test, y_train, y_test = train_test_split(biomarkers[k], adhd_labels, test_size=0.2, shuffle=True)
        classifier.fit(X_train, y_train)
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
    masker = get_atlas_data(map)

    adhd_data = generate_train_data(1.45)
    adhd_subjects, pooled_subjects, site_names, o_adhd_labels = get_adhd_dataset_info(adhd_data, masker)

    # kinds = ['correlation', 'partial correlation', 'tangent', 'covariance', 'precision']
    kinds = ['tangent']
    biomarkers, adhd_labels = make_connectivity_biomarkers(kinds, o_adhd_labels, adhd_data, adhd_data.func,
                                                           pooled_subjects)

    accuracies = run_model(kinds, biomarkers, adhd_labels, layers, graph=True,
                           c_matrix=True, print_results=True, f1=True)
    accuracies = accuracies.values()
    print 'accuracies', accuracies
    print 'mean', np.mean(accuracies)
    print 'std', np.std(accuracies)
    print "ran model", time.time() - t0, 'seconds'
    return accuracies


if __name__ == '__main__':
    main('sub-maxprob-thr50-2mm', 462)
    #main('sub-maxprob-thr50-2mm', 136)

'''
def main(layers, kinds, biomarkers, adhd_labels):
    accuracies = run_model(kinds, biomarkers, adhd_labels, layers, graph=True,
                           c_matrix=True, print_results=True, f1=True)
    return accuracies


if __name__ == '__main__':
    t0 = time.time()
    masker = get_atlas_data('sub-maxprob-thr50-2mm')

    adhd_data = generate_train_data(1.45)
    adhd_subjects, pooled_subjects, site_names, adhd_labels = get_adhd_dataset_info(adhd_data, masker)
 
    # kinds = ['correlation', 'partial correlation', 'tangent', 'covariance', 'precision']
    kinds = ['tangent']
    biomarkers = make_connectivity_biomarkers(kinds, pooled_subjects)
    layers = [136, 462]
    mean_f1s = {}
    for layer in layers:
        cv_scores = []
        for i in range(500):
            t0 = time.time()
            accuracies = main(layer, kinds, biomarkers, adhd_labels)
            cv_scores.append(accuracies['tangent'])
            print 'ran layer ({0}) iter {1} in {2} seconds'.format(layer, i, time.time() - t0)
        print "= " * 20
        # mean_f1s[(layer, layer2)] = [np.mean(cv_scores), np.std(cv_scores), max(cv_scores), min(cv_scores)]
        mean_f1s[layer] = {'mean': np.mean(cv_scores), 'std': np.std(cv_scores), 'max': max(cv_scores),
                           'min': min(cv_scores)}
        pprint.pprint(mean_f1s)
        print "= " * 20
    pprint.pprint(mean_f1s)

    # with tts as 0.2
'''
