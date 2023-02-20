# Project 1 - 02450 Introduction to Machine Learning and Data Mining
The objective of this report is to apply the methods learned in the first section of the course, ”Data: Feature extraction, and visualization” on a data set for spam emails, in order to get a basic understanding of our data prior to the further analysis which will be performed in project 2.

### Data Set
The data set was obtained from the UC Irvine Machine Learning Repository. The specific data set can be downloaded from the following link https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/.
The uncompressed zip folder can be directly added to the working directory, with the functions.get_spam_data() taking care of processing the files in said directory to create a dataframe to use throughout the project. Note that the specific path to the "spambase" folder must be passed as a string parameter to the previously mentioned function.

### Directory Structure
The functions.py file contains all functions created for later use in the main.py file