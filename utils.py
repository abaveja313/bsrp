import fnmatch
import os
import subprocess
import re
import pandas as pd
import random
import numpy as np
import progressbar
from sklearn.model_selection import GridSearchCV


class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ADHD200(object):
    def __init__(self):
        self.func = []
        self.ids = []
        self.test_ids = []
        self.pheno = []
        self.test_func = []
        self.test_pheno = []
        self.test_site_names = []
        self.test_labels = []
        self.root_dir = '/Volumes/Amrit\'s SSD'
        # self.root_dir = '/home/ec2-user/data'
        self.train_glob = 'sfnwmrda*_session_*_rest_1.nii'
        self.id_regex = '.*?\/([0-9].*?)\/'
        self.pheno_regex = '.*?\/train/(.*?)\/'
        self.test_pheno_regex = '.*?\/test/(.*?)\/'
        self.pheno_filename = 'phenotypic.csv'
        self.labels = []
        self.site_names = []

    def _find_niis(self, train=True):
        if train:
            dr = '/train'
        else:
            dr = '/test'

        for root, dirnames, filenames in os.walk(self.root_dir + dr):
            for filename in fnmatch.filter(filenames, self.train_glob):
                file = os.path.join(root, filename)
                if train:
                    prefix = re.search(self.pheno_regex, file).group(1)
                    self.func.append(file)
                    self.ids.append(prefix + '_' + re.search(self.id_regex, file).group(1).lstrip('0'))
                    self.site_names.append(prefix)
                else:
                    prefix = re.search(self.test_pheno_regex, file).group(1)
                    self.test_func.append(file)
                    self.test_ids.append(prefix + '_' + re.search(self.id_regex, file).group(1).lstrip('0'))
                    self.test_site_names.append(prefix)

    def unzip_files(self, dir, train=True):
        if train:
            dr = '/train/'
        else:
            dr = '/test/'

        for root, dirnames, filenames in os.walk(self.root_dir + dr + dir):
            for filename in fnmatch.filter(filenames, 'sfnwmrda*_session_*_rest_1.nii.gz'):
                file_name = os.path.join(root, filename)
                subprocess.call(['gunzip', '-k', file_name])

    def _add_phenotypic(self, pheno_filename, id, train=True):
        df = pd.read_csv(self.root_dir + '/' + pheno_filename)
        if train:
            self.pheno.append(df[df['Subject'] == id].values.tolist())
        else:
            self.test_pheno.append(df[df['Subject'] == id].values.tolist())
        return [df[df['Subject'] == id], df[df['Subject'] == id]['Group']]

    def gen_data(self, train=True):
        temp_func = []
        temp_pheno = []
        temp_sites = []

        if train:
            labels = self.labels
            ids = self.ids
            self._find_niis()
            ops = 0
            bar = progressbar.ProgressBar(max_value=len(self.ids))
            for index, id in enumerate(ids):
                ops += 1
                bar.update(ops)
                res = self._add_phenotypic(self.pheno_filename, id)
                if res[0]['Group'].empty:
                    continue
                else:
                    temp_func.append(self.func[index])
                    temp_pheno.append(self.pheno[index])
                    temp_sites.append(self.site_names[index])
                    if 'ADHD' in res[1].tolist()[0]:
                        labels.append(1)

                    else:
                        labels.append(0)
        else:
            labels = self.test_labels
            ids = self.test_ids
            self._find_niis(train=False)
            ops = 0
            bar = progressbar.ProgressBar(max_value=len(self.test_ids))
            for index, id in enumerate(ids):
                ops += 1
                bar.update(ops)
                res = self._add_phenotypic(self.pheno_filename, id, train=False)
                if res[0]['Group'].empty:
                    continue
                else:
                    temp_func.append(self.test_func[index])
                    temp_pheno.append(self.test_pheno[index])
                    temp_sites.append(self.test_site_names[index])
                    if 'ADHD' in res[1].tolist()[0]:
                        self.test_labels.append(1)

                    else:
                        self.test_labels.append(0)
        if train:
            self.func = temp_func
            self.pheno = temp_pheno
            self.site_names = temp_sites
            self.labels = labels
            bar.finish()

            # self.equilize_data(eq)
        else:
            self.test_func = temp_func
            self.test_pheno = temp_pheno
            self.test_site_names = temp_sites
            self.test_labels = labels
            bar.finish()

    def equilize_data(self, ratio):
        samples = list(zip(self.func, self.pheno, self.labels, self.site_names))
        random.shuffle(samples)
        func, pheno, labels, site_names = zip(*samples)
        temp_func = []
        temp_pheno = []
        temp_sites = []
        temp_labels = []
        adhd_samples = len([i for i in labels if i == 1])
        non_adhd_samples = 0
        for index in range(len(self.func)):
            if adhd_samples * ratio <= non_adhd_samples:
                if labels[index] == 1:
                    temp_func.append(func[index])
                    temp_pheno.append(pheno[index])
                    temp_sites.append(site_names[index])
                    temp_labels.append(labels[index])
                else:
                    continue
            else:
                non_adhd_samples += 1
                temp_func.append(func[index])
                temp_pheno.append(pheno[index])
                temp_sites.append(site_names[index])
                temp_labels.append(labels[index])

            self.func = temp_func
            self.pheno = temp_pheno
            self.site_names = temp_sites
            self.labels = temp_labels

    def retrieve_pheno_for_model(self, index):
        if isinstance(index, int):
            pheno = pd.read_csv(self.root_dir + '/train/train_pheno/' + self.site_names[index] + '/phenotypic_comb.csv')
            scanDirId = self.ids[index].split('_')[1]
            subject = pheno[pheno['ScanDirID'] == int(scanDirId)]
        else:
            ind = self.func.index(index)
            pheno = pd.read_csv(self.root_dir + '/train/train_pheno/' + self.site_names[ind] + '/phenotypic_comb.csv')
            scanDirId = self.ids[ind].split('_')[1]
            subject = pheno[pheno['ScanDirID'] == int(scanDirId)]
        res = [subject['Age'].values[0], subject['Gender'].values[0], subject['Handedness'].values[0]]
        if np.isnan(res).any():
            return False
        else:
            return res

    def gen_pheno(self):
        X = []
        y = []
        length = len(self.func)
        bar = progressbar.ProgressBar(max_value=length)
        ops = 0
        for index in range(length):
            res = self.retrieve_pheno_for_model(index)
            ops += 1
            bar.update(ops)
            if res:
                X.append(res)
                y.append(self.labels[index])

        bar.finish()
        X = np.array(X)
        X.reshape(-1, 1)
        y = np.array(y)
        y.reshape(-1, 1)
        return X, y


class TestADHD200:
    def __init__(self, adhd200Obj):
        self.func = adhd200Obj.test_func
        self.pheno = adhd200Obj.test_pheno
        self.labels = adhd200Obj.test_labels
        self.site_names = adhd200Obj.test_site_names


def conform_1d(vals, target):
    new_array = [0] * target
    for i in range(0, len(vals)):
        new_array[i] = vals[i]
    return np.array(new_array)


fieldnames = ['mlp_layesr', 'mlp_solver', 'logistic_regressions', 'svc_kernel', 'xgb_estimators',
              'early_stopping_iterations', 'positions', 'mean_accuracy', 'max_accuracy', 'min_accuracy',
              'std_accuracy', 'mean_f1', 'max_f1', 'min_f1', 'std_f1', 'mean_precision', 'max_precision',
              'min_precision', 'std_precision', 'mean_recall', 'max_recall', 'min_recall', 'std_recall']


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


# TODO RUN PARAM OPTIMZER FOR F1-SCORE
def get_gc_config(mlp_layer, mlp_solver, number_logistic, svc_kernel, num_xgb_estim, early_stopping_rounds, positions):
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 1000
    ca_config["early_stopping_rounds"] = early_stopping_rounds
    ca_config["n_classes"] = 2
    estimators_to_append = []
    ca_config["estimators"] = [0] * 6

    if mlp_layer and mlp_solver:
        estimators_to_append.append(
            {
                "n_folds": 4,
                "type": "MLPClassifier",
                "hidden_layer_sizes": mlp_layer,
                "random_state": 0,
                'solver': mlp_solver
            }
        )

    for _ in range(number_logistic[0]):
        estimators_to_append.append({"n_folds": 4, "type": "LogisticRegression"})

    if num_xgb_estim:
        estimators_to_append.append(
            {"n_folds": 5,
             "type": "XGBClassifier",
             "n_estimators": num_xgb_estim,
             "objective": "multi:softprob",
             "silent": True,
             "nthread": -1,
             "num_class": 2,

             }
        )

    for _ in range(number_logistic[1]):
        estimators_to_append.append({"n_folds": 4, "type": "LogisticRegression"})

    if svc_kernel:
        estimators_to_append.append(
            {
                "n_folds": 4,
                "type": "SVC",
                "random_state": 0,
                "kernel": type,
                "probability": True
            }
        )

    estimators_to_append.append({"n_folds": 4, "type": "LogisticRegression"})
    if positions:
        for index, value in enumerate(estimators_to_append):
            ca_config['estimators'][positions.index(index)] = value
    else:
        ca_config['estimators'] = estimators_to_append

    print 'Using config', ca_config

    config["cascade"] = ca_config
    return config


'''
    ca_config["estimators"].append( # 5 rf estimators gives 73%?
        {"n_folds": 5,
         "type": "RandomForestClassifier",
         "n_estimators": est1,  # MODEK_PHENO1=50, MODEL_PHENO2 = 256
         "max_depth": None,
         "n_jobs": -1,
         }
    )
      ca_config["estimators"].append(
        {"n_folds": 5,
         "type": "XGBClassifier",
         "n_estimators": d,
         "objective": "multi:softprob",
         "silent": True,
         "nthread": -1,
         "num_class": 2,

         }
    )
'''
