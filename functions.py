import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def get_spam_data(data_dir):
    '''
    Function to create a dataframe from the spam data set, by cleaning up and merging the values and the attribute names
    '''
    
    # Specify paths of the data and names, given the path of the spam folder
    file_path_data = data_dir + "/spambase.data"
    file_path_names = data_dir + "/spambase.names"

    # Get attributes from spambase.names by ignoring content that does not include the 57 attribute names (located at the last 57 lines)
    spam_names_messy = open(file_path_names).readlines()
    first_line = len(spam_names_messy) - 57
    last_line = len(spam_names_messy)
    spam_names_messy = spam_names_messy[first_line:last_line]

    # Clean up lines and create array with attributes
    spam_names = []
    for i in range(len(spam_names_messy)):
        spam_names.append(spam_names_messy[i].replace("continuous.\n", "").translate({ord(i): None for i in ': '}))

    # Add spam classification column to attribute names
    spam_names.append("spam_class")

    # Create dataframe with data and attribute names
    spam_data = pd.read_csv(file_path_data, header=None)
    spam_data.columns = spam_names


    return spam_data

