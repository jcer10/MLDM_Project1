# Project 1 - 02450 Introduction to Machine Learning and Data Mining
The objective of this report is to apply the methods learned in the first section of the course, ”Data: Feature extraction, and visualization” on a data set for spam emails, in order to get a basic understanding of our data prior to the further analysis which will be performed in project 2.

### Data Set
The data set was obtained from the UC Irvine Machine Learning Repository. The specific data set can be downloaded from the following link https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/. The uncompressed zip folder can be directly added to the working directory.

The function "get_df()" is in charge of creating a pandas dataframe from the files inside the spambase folder (the spambase.data and spambase.names). The function df_to_arrays creates a python dict containing the standard representation variables used in the course, including: X, attributes, N, M 
