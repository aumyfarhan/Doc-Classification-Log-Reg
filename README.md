# Doc-Classification-Log-Reg

README
Program: Logistic Regression Classification
Performs estimation of the parameters of a “Logistic Regression Classification” model using training data and then can be used for classification of new unclassified data.

Program Description: 
Takes input one training file and one testing file in CSV format (of specific attribute, attribute values and class label). Estimates the parameters using the training data to be used for classification of the testing data. 

Running The Program:
We have provided a python script named ‘logistic_regression.py’ which can be run from command line in any operating system. To run the script first make the folder with the python script ‘logistic_regression.py’ as the current folder. Also, make sure that both the training file and testing file are present in the same folder.

Requirements:
Needs python version 3.6 or higher available to run the script.

Has dependency on the following python modules:

pandas ( to read csv files)
numpy (for array manipulation)
math (to calculate logarithmic values)
csv (to write into csv files)
matplotlib (for plotting)


Usage:
An example command line input to run the script:

python logistic_regression.py

Here first argument ‘python’ asks to use python for compiling, ‘logistic_regression.py’ is the python script name.

Usage Options:
A generic command line input to run the script:

python script_name

script_name: name of the python script (here, ‘logistic_regression.py’)

Built With:

Python version 3.6.2
