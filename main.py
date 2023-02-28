from functions import *


# Create dataframe by specifying folder path of data
spam_df = get_df("./spambase")

# Create dict with standard data format
spam_data = df_to_arrays(spam_df)

pc_variance_plot(spam_data)

pc_data_plot(spam_data)

# Create df with basic summary statistics from data set
summary_stats = spam_df.describe().drop("count")
