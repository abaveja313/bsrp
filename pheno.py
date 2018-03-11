import numpy as np
from utils import ADHD200
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier

# best is extra trees
def get_model( tts=0.2):
    adhd200 = ADHD200()
    adhd200.gen_data()
    X, y = adhd200.gen_pheno()
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tts)
    print 'training = {0} samples\ntesting = {1} samples'.format(len(X_train), len(y_test))

    mclf = ExtraTreesClassifier(n_estimators=25, warm_start=True)
    clf = BaggingClassifier(base_estimator=mclf, verbose=8, n_estimators=750, random_state=0)
    clf.fit(X_train, y_train)

    print 'Pheno Classification Accuracy:'
    y_pred = clf.predict(X_test)
    print 'pred', y_pred
    print 'true', y_test
    print classification_report(y_test, y_pred)
    print 'accuracy', accuracy_score(y_test, y_pred)
    return clf


def predict(clf, adhd200, func_file):
    X = adhd200.retrieve_pheno_for_model(func_file)
    print X
    if X:
        probs = clf.predict_proba([X])
        out = clf.predict([X])
        if out == 1:
            return probs[0][1]
        else:
            return probs[0][0]
    else:
        return False

'''
import time
import pprint
import numpy as np

# y_pred = clf.predict_proba(X_test)
adhd200 = ADHD200()
adhd200.gen_data()
X, y = adhd200.gen_pheno()
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

layers = [106]
mean_f1s = {}
for layer in layers:
    cv_scores = []
    for i in range(20):
        t0 = time.time()
        accuracy = get_model(X, y, 30, layer)
        cv_scores.append(accuracy)
        print 'ran estim ({0}) iter {1} in {2} seconds'.format(layer, i, time.time() - t0)
    print "= " * 20
    # mean_f1s[(layer, layer2)] = [np.mean(cv_scores), np.std(cv_scores), max(cv_scores), min(cv_scores)]
    mean_f1s[layer] = {'mean': np.mean(cv_scores), 'std': np.std(cv_scores), 'max': max(cv_scores),
                       'min': min(cv_scores)}
    d = sorted(mean_f1s.iteritems(), key=lambda (x, y): y['mean'])
    d.reverse()
    pprint.pprint(d)
    print "= " * 20
d = sorted(mean_f1s.iteritems(), key=lambda (x, y): y['mean'])
d.reverse()
pprint.pprint(d)
'''