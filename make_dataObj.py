import fnmatch
import os
import subprocess
import re
import pandas as pd
import random


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
                print "file", file_name
                subprocess.call(['gunzip', '-k', file_name])

    def _add_phenotypic(self, pheno_filename, id, train=True):
        df = pd.read_csv(self.root_dir + '/' + pheno_filename)
        if train:
            self.pheno.append(df[df['Subject'] == id].values.tolist())
        else:
            self.test_pheno.append(df[df['Subject'] == id].values.tolist())
        return [df[df['Subject'] == id], df[df['Subject'] == id]['Group']]

    def gen_data(self, eq=1, train=True):
        temp_func = []
        temp_pheno = []
        temp_sites = []

        if train:
            labels = self.labels
            ids = self.ids
            self._find_niis()
            for index, id in enumerate(ids):
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
            for index, id in enumerate(ids):
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
            #self.equilize_data(eq)
        else:
            self.test_func = temp_func
            self.test_pheno = temp_pheno
            self.test_site_names = temp_sites
            self.test_labels = labels

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


class TestADHD200:
    def __init__(self, adhd200Obj):
        self.func = adhd200Obj.test_func
        self.pheno = adhd200Obj.test_pheno
        self.labels = adhd200Obj.test_labels
        self.site_names = adhd200Obj.test_site_names

'''
a = ADHD200()
a.unzip_files('NYU')
a.unzip_files('OHSU')
a.unzip_files('Peking')
a.unzip_files('Pittsburgh')
a.unzip_files('WashU')
a.unzip_files('KKI')
a.unzip_files('NeuroIMAGE')
a.unzip_files('KKI', train=False)
a.unzip_files('NeuroIMAGE', train=False)
a.unzip_files('OHSU', train=False)
a.unzip_files('Peking', train=False)
'''