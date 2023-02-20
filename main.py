from functions import *

# Create dataframe by specifying folder path of data
spam_data = get_spam_data("./spambase")

# Check that the dataframe is correct
print(spam_data)