from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from utils import ADHD200
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# best is extra trees
def get_model(tts=0.2):
    adhd200 = ADHD200()
    adhd200.gen_data()
    X, y = adhd200.gen_pheno()
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tts)
    print 'training = {0} samples\ntesting = {1} samples'.format(len(X_train), len(y_test)
                                                                 )
    # clf = GaussianNB()
    mclf = ExtraTreesClassifier(n_estimators=25, warm_start=True)
    # vclf = VotingClassifier(estimators=[('gnb', gnb), ('etc', etc)], voting='soft', weights=[1.675, 1])
    clf = BaggingClassifier(base_estimator=mclf, verbose=8, n_estimators=750, random_state=0)
    clf.fit(X_train, y_train)

    print 'Pheno Classification Accuracy:'
    y_pred = clf.predict(X_test)

    print 'pred', y_pred
    print 'true', y_test
    print '\naccuracy:', accuracy_score(y_test, y_pred)
    print classification_report(y_test, y_pred)

    return clf


def predict(clf, adhd200, func_file):
    X = adhd200.retrieve_pheno_for_model(func_file)
    if X:
        probs = clf.predict_proba([X])
        out = clf.predict([X])
        if out == 1:
            return probs[0][1]
        else:
            return probs[0][0]
    else:
        return False


# y_pred = clf.predict_proba(X_test)
