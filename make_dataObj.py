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
        self.pheno = []
        self.fp_func = []
        self.fp_pheno = []
        self.fp_site_names = []
        self.fp_labels = []
        self.root_dir = '/Volumes/Amrit\'s SSD'
        self.train_glob = 'sfnwmrda*_session_*_rest_1.nii'
        self.id_regex = '.*?\/([0-9].*?)\/'
        self.pheno_regex = '.*?\/train/(.*?)\/'
        self.pheno_filename = 'phenotypic.csv'
        self.labels = []
        self.site_names = []

    def _find_niis(self):
        for root, dirnames, filenames in os.walk(self.root_dir + '/train'):
            for filename in fnmatch.filter(filenames, self.train_glob):
                file = os.path.join(root, filename)
                self.func.append(file)
                prefix = re.search(self.pheno_regex, file).group(1)
                self.ids.append(prefix + '_' + re.search(self.id_regex, file).group(1).lstrip('0'))
                self.site_names.append(prefix)
        print self.func
        print self.ids

    def _unzip_files(self, dir):
        for root, dirnames, filenames in os.walk(self.root_dir + '/train/' + dir):
            for filename in fnmatch.filter(filenames, 'sfnwmrda*_session_*_rest_*.nii.gz'):
                file_name = os.path.join(root, filename)
                print "file", file_name
                subprocess.call(['gunzip', '-k', file_name])

    def _add_phenotypic(self, pheno_filename, id):
        df = pd.read_csv(self.root_dir + '/' + pheno_filename)
        self.pheno.append(df[df['Subject'] == id].values.tolist())
        return [df[df['Subject'] == id], df[df['Subject'] == id]['Group']]

    def gen_data(self):
        self._find_niis()
        temp_func = []
        temp_pheno = []
        temp_sites = []
        for index, id in enumerate(self.ids):
            res = self._add_phenotypic(self.pheno_filename, id)

            if res[0]['Group'].empty:
                continue
            else:
                temp_func.append(self.func[index])
                temp_pheno.append(self.pheno[index])
                temp_sites.append(self.site_names[index])
                if 'ADHD' in res[1].tolist()[0]:
                    self.labels.append(1)

                else:
                    self.labels.append(0)
        self.func = temp_func
        self.pheno = temp_pheno
        self.site_names = temp_sites
        self.equilize_data(1.75)

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
            if adhd_samples * ratio == non_adhd_samples:
                if labels[index] == 0:
                    self.fp_func.append(func[index])
                    self.fp_pheno.append(pheno[index])
                    self.fp_site_names.append(site_names[index])
                    self.fp_labels.append(labels[index])
                else:
                    temp_func.append(func[index])
                    temp_pheno.append(pheno[index])
                    temp_sites.append(site_names[index])
                    temp_labels.append(labels[index])
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


class FPADHD200:
    def __init__(self, adhd200Obj):
        self.func = adhd200Obj.fp_func
        self.pheno = adhd200Obj.fp_pheno
        self.labels = adhd200Obj.fp_labels
        self.site_names = adhd200Obj.fp_site_names


