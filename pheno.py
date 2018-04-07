import numpy as np
from utils import ADHD200, conform_1d
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier


# best is extra trees
def get_model(tts=0.2):
    adhd200 = ADHD200()
    adhd200.gen_data()
    X, y = adhd200.gen_pheno()
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tts)
    print 'training = {0} samples\ntesting = {1} samples'.format(len(X_train), len(y_test))
    # clf = DBN() #50 #11
    mclf = ExtraTreesClassifier(n_estimators=30, warm_start=True)
    clf = BaggingClassifier(base_estimator=mclf, verbose=4, n_estimators=125, random_state=0)
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
        probs = clf.predict_proba(np.array([X]))
        out = clf.predict(np.array([X]))
        print out
        if out == 1:
            return probs[0][1]
        else:
            return -1 * probs[0][0]
    else:
        return False

def get_params(adhd200, func_file):
    X = adhd200.retrieve_pheno_for_model(func_file)
    if X:
        return X
    return None
