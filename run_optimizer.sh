#!/usr/bin/env bash
# Install dependencies in a virtualenv and run the optimizer
# This is for running on the Amazon Linux AMI. For OSX, simply remove the yum install command and
# if you don't already have them, install the xcode command line tools (xcode-select --install)
# Run this as root, (using sudo) so that the script has the necessary permissions to run

echo "Installing virtual environment package"
pip install virtualenv # install the virtual environment package
echo "Setting up virtual environment"
virtualenv adhd_diagnosis_env # setup a virtual environment
echo "Activating environment"
source adhd_diagnosis_env/bin/activate # activate it
echo "installing compilers for c and c++"
yum install gcc gcc-c++ # install compilers for xgboost
echo "installing python packages"
pip install -r requirements.txt # Install package requirements
echo "Make sure you have the data downloaded and formatted correctly, as specified in the README.md file"
echo "Running the optimizer and exporting results to results.csv. Type ctrl-c to abort"
python main.py # Run the optimizer python script


