# COMP 4900: MINI PROJECT 1 
## Linear Classifier using Logistic Regression 
### Team Members - Group 1-3
Max Polzin \
Keyanna Coghlan

### Overview
This project implements a logistic regression classifer model with k-folds validation and various other scripts to evaulate the on two datasets.
The following intructions detail how to reproduce the results discussed in the acompying report by running
the provided code in Colabs or in your own local desktop environment.

## Colabs
### Setting up the Colabs Environment
1. Unzip the provided file called "LogisticRegression.zip".
2. Upload this folder into a Google Drive
3. Upload the Google Colabs file called "COMP4900-A1.py" (or "COMP4900-A1.ipynb") to the same Google Drive

### Excuting in Colabs
1. Run "Step 1" in the Google Colabs file. This will mount the drive that contains the "LogisticRegression" folder.
2. Run "Step 2" in the Google Colabs file. This will generate graphs and other data that was used in the 
statistical analysis presented in the accompying report.
3. Run "Step 3" in the Google Colabs file. This will train, validate and test the Logistic Regression model on the 
two datasets that are located in the "datasets" folder inside of the "LogisticRegression" folder.

## Locally
### Setting up the local Environment
1. Unzip the provided file called "LogisticRegression.zip".
2. Install Python 3.6 or greater and Pip 19.2.3
3. Pip install the following libraries: Pandas 1.1.2, Numpy 1.19.1, Matplotlib 3.3.0

### Excuting locally
1. Open a terminal and cd into the "LogisticRegression" folder.
2. Run: "python scripts/statAnalysis.py" \
This will generate graphs and other data that was used in the 
statistical analysis presented in the accompying report.     
4. Run: "python models/modelTesting.py" \
This will train, validate and test the Logistic Regression model on the 
two datasets that are located in the "datasets" folder inside of the "LogisticRegression" folder.