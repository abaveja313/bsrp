# Using the ADHD200 dataset to diagnose ADHD using gcForest
This is the code to accompany the paper "Machine Learning Alternatives for ADHD Diagnosis using fMRI volumes and phenotypic information" by Amrit Baveja in 2018
### It was created as an entry into the Branson Science Research Project Symposium
## Usage
On OS X, remove the `yum install gcc gcc-c++` from the `run_optimizer.sh`. If you haven't already installed the Xcode command line tools, install them using `xcode-select --install` (installs C/C++ compilers needed for xgboost).

```
cd <path to BSRP git repository>
screen # if you want to run the optimizer in the background
chmod +x run_optimizer.sh
sudo ./run_optimizer.sh
```

## Description
The code in this repository runs a randomized optimizer that tests different combinations of scikit-learn classifiers with random hyperparameters in the structure specified by the [gcForest paper](https://arxiv.org/abs/1702.08835).
The gcForest implementation is from [https://github.com/kingfengji/gcForest](https://github.com/kingfengji/gcForest).

I extended the gcForest implementation to use these classifiers
1. Multi Layer Perceptron
2. Random Forests
3. Logistic Regression
4. SVC (linear, rbf, polynomial)
5. XGBoost Classifier

Using these random combinations of these classifiers on a randomly generated test dataset (20% by default) over a specified number of cross validation iterations, it logs a variety of metrics to a CSV file for later analysis.
1. F1 Score (mean, max, min, standard deviation)
2. Confusion matrix (mean, max, min, standard deviation)
3. Accuracy (mean, max, min, standard deviation)
4. Precision (mean, max, min, standard deviation)
5. Recall (mean, max, min, standard deviation)

Using spreadsheet software such as Numbers, Google Sheets or Excel, you can easilly load the CSV file and sort to find the ideal layer combination. 

### All of the parameters you can tweak are inside the `__init__` method of the `Optimizer` class (main.py)
