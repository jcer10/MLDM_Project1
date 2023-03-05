from functions import *


# Create dataframe by specifying folder path of data
spam_df = get_df("./spambase")

# Create dict with standard data format
spam_data = df_to_arrays(spam_df)

pc_variance_plot(spam_data)

pc_data_plot(spam_data)


# Create df with basic summary statistics from data set
summary_stats = spam_df.describe().drop("count")



#Create boxplot
#box_plot(spam_data)
#Boxplot of the last 3 attributes
#box_plot_last_attributes(spam_data)
#Boxplot with standardized data
#box_plot_standardized(spam_data)




#Correlation matrix
correlation_heatmap(spam_data)

#The coefficients of the first 3 components for each attribute
pca_coefffs(spam_data)
#Plot histograms for all the attributes
attributes_histogram(spam_data)
