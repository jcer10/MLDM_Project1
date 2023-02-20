from functions import *

# Create dataframe by specifying folder path of data
spam_df = get_df("./spambase")

# Create dict with standard data format
spam_data = df_to_arrays(spam_df)
