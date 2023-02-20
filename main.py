from functions import *

# Create dataframe by specifying folder path of data
spam_data = get_dataframe("./spambase")

# Check that the dataframe is correct
print(spam_data)