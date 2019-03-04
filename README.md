# Randomized Searching algorithm to optimize gcForest for ADHD diagnosis
This is the code to accompany the paper "Machine Learning Alternatives for ADHD Diagnosis using fMRI volumes and phenotypic information"


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1227447.svg)](https://doi.org/10.5281/zenodo.1227447)

### It was created as an entry into the 9th Grade Branson Science Research Project Symposium
## Usage
The `run_optimizer.sh` file is meant for running on Amazon Linux (AWS) or a Centos 6/7 like Linux variant. To run on OS X, remove the `yum install gcc gcc-c++` from the `run_optimizer.sh` file. If you haven't already installed the Xcode command line tools, install them using `xcode-select --install` (installs C/C++ compilers needed for xgboost). Additionally, you need to specify a path to the Athena pipeline preprocessed ADHD200 data from the NITRC website in the `utils.py` file.

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

Using spreadsheet software such as Numbers, Google Sheets or Excel, you can easily load the CSV file and sort/filter to find the ideal layer combination. 

#### All of the parameters you can adjust are inside the `__init__` method of the `Optimizer` class (main.py)
