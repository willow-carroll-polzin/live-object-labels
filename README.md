# MAAJ 5308: FINAL PROJECT
## Semantic Map Labelling
### Team Members
Max Polzin

### Overview
This project implements a logistic regression classifer model with k-folds validation and various other scripts to evaulate the on two datasets.
The following intructions detail how to reproduce the results discussed in the acompying report by running
the provided code in Colabs or in your own local desktop environment.

## Dependencies:
This project uses ????? YOLOv3 ..

## Setup and Usage:
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