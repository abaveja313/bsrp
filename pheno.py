from sklearn.naive_bayes import GaussianNB
from utils import ADHD200
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def get_model(tts=0.2):
    adhd200 = ADHD200()
    adhd200.gen_data()
    X, y = adhd200.gen_pheno()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tts)
    print 'training = {0} samples\ntesting = {1} samples'.format(len(X_train), len(y_test))
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    #y_pred = clf.predict(X_test)
    #print classification_report(y_test, y_pred)
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

#model, adhd200 = get_model(tts=0.2)

#print predict(model, adhd200, '/Volumes/Amrit\'s SSD/train/NYU/0010007/sfnwmrda0010007_session_1_rest_1.nii')