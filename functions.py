import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend,boxplot,bar,xticks,grid,subplot,hist
import seaborn as sns

from scipy.stats import zscore

def get_df(data_dir):
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
    spam_df = pd.read_csv(file_path_data, header=None)
    spam_df.columns = spam_names

    return spam_df


def df_to_arrays(df):
    '''
    Function that converts the data frame into the standard form data and stores the values in a dict
    '''

    # Extract values from df in matrix form
    raw_data = df.values

    # Separate last column that correspond to the class index (y vector) from the data matrix (X vector)
    attributes = len(raw_data[0])
    cols = range(0, attributes - 1)

    X = raw_data[:, cols]
    y = raw_data[:, -1]
    # Obtain dimensions of data matrix
    N, M = X.shape

    # Extract the attribute names from df
    attributeNames = np.asarray(df.columns[cols])

    # Define class names and number of classes
    classNames = ["non_spam", "spam"]
    C = len(classNames)

    # Create data dict
    data_dict = {"X": X, "attributeNames": attributeNames, "N": N, "M": M, "y": y, "classNames": classNames, "C": C}

    return data_dict


def pc_variance_plot(data_dict):
    # test
    # Subtract mean value from data
    Y = (data_dict["X"] - np.ones((data_dict["N"], 1)) * data_dict["X"].mean(axis=0)) / data_dict["X"].std(axis=0)

    # PCA by computing SVD of Y
    U, S, V = svd(Y, full_matrices=False)

    # Compute variance explained by principal components
    rho = (S * S) / (S * S).sum()

    threshold = 0.9

    # Plot variance explained
    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, 'x-')
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
    plt.plot([1, len(rho)], [threshold, threshold], 'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual', 'Cumulative', 'Threshold'])
    plt.grid()
    plt.show()


def pc_data_plot(data_dict):
    # Subtract mean value from data
    Y = (data_dict["X"] - np.ones((data_dict["N"], 1)) * data_dict["X"].mean(axis=0)) / data_dict["X"].std(axis=0)

    # PCA by computing SVD of Y
    U, S, Vh = svd(Y, full_matrices=False)
    # scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
    # of the vector V. So, for us to obtain the correct V, we transpose:
    V = Vh.T
    # Project the centered data onto principal component space
    Z = Y @ V

    # Indices of the principal components to be plotted
    i = 0
    j = 1

    # Plot PCA of the data
    f = figure()
    title('Spam data: PCA')
    # Z = array(Z)
    for c in range(len(data_dict["attributeNames"])):
        # select indices belonging to class c:
        class_mask = data_dict["y"] == c
        plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.5)
    legend(data_dict["classNames"])
    xlabel('PC{0}'.format(i + 1))
    ylabel('PC{0}'.format(j + 1))

    # Output result to screen
    show()



# Attributes boxplot
def box_plot(data_dict):
    data = data_dict['X']
    figure(figsize=(12,6))
    title('Boxplot')
    boxplot(data)
    xlabel("Attributes")
    show()
    
    


#Box plot for only the last 3 attributes
def box_plot_last_attributes(data_dict):
    data = data_dict['X']
    last_attributes=data[:,54:57]
    figure()
    title('Boxplot of the last 3 attributes')
    boxplot(last_attributes)
    xlabel("Attributes")
    show()

#boxplot with standardized data
def box_plot_standardized(data_dict):
    figure(figsize=(12, 6))
    title('Boxplot (standardized)')
    xlabel("Attributes")
    data = (data_dict["X"] - np.ones((data_dict["N"], 1)) * data_dict["X"].mean(axis=0)) / data_dict["X"].std(axis=0)
    boxplot(data)
    show()


def correlation_heatmap(data_dict):
    correlation_data = {}
    for i in range(len(data_dict['attributeNames'])):
        correlation_value = []
        for j in range(len(data_dict['attributeNames'])):
            pccs = np.corrcoef(list(data_dict['X'][:,i]),list(data_dict['X'][:,j]))[0][1]
            correlation_value.append(float(pccs))
            if abs(pccs) >= 0.99 or abs(pccs) < 0.5:
                continue

        correlation_data[data_dict['attributeNames'][i]] = correlation_value

    df = pd.DataFrame(correlation_data).corr()
    ax = sns.heatmap(df)
    ax.set_xticks(np.arange(0.5, len(df.columns), 5))
    ax.set_xticklabels(np.arange(0, len(df.columns)+1, 5))
    ax.set_yticks(np.arange(0.5, len(df.columns), 5))
    ax.set_yticklabels(np.arange(0, len(df.columns)+1, 5))
    title('Correlation heatmap between feauters')
    show()


# Display the components coefficients for the first 3 components
def pca_coefffs(data_dict):
    # Normalize data
    Y = (data_dict["X"] - np.ones((data_dict["N"], 1)) * data_dict["X"].mean(axis=0)) / data_dict["X"].std(axis=0)

    # PCA by computing SVD of Y
    U, S, Vh = svd(Y, full_matrices=False)
    # scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
    # of the vector V. So, for us to obtain the correct V, we transpose:
    V = Vh.T
    # Project the centered data onto principal component space
    Z = Y @ V

    #choose the first 3 components
    components = [0, 1, 2]

    legend_stings = ['PC' + str(e + 1) for e in components]
    bar_width = .2

    #first 48 attributes
    rang = np.arange(1, len(data_dict['attributeNames'][:48]) + 1)
    for i in components:
        bar(rang + i * bar_width, V[:48, i], width=bar_width)
    #xticks(rang + bar_width, data_dict['attributeNames'][:48], rotation=90)
    xticks(rang + bar_width, [i+1 for i in range(48)], rotation=0)
    xlabel('Attributes')
    ylabel('Component coefficients values')
    legend(legend_stings)
    grid()
    title('PCA Components Coefficients')
    show()

    #next 6 attributes
    rang = np.arange(1, len(data_dict['attributeNames'][48:54]) + 1)
    for i in components:
        bar(rang + i * bar_width, V[48:54, i], width=bar_width)
    xticks(rang + bar_width, data_dict['attributeNames'][48:54], rotation=90)
    xlabel('Attributes')
    ylabel('Component coefficients values')
    legend(legend_stings)
    grid()
    title('PCA Components Coefficients')
    show()

    #last 3 attributes
    rang = np.arange(1, len(data_dict['attributeNames'][-3:]) + 1)
    for i in components:
        bar(rang + i * bar_width, V[-3:, i], width=bar_width)
    xticks(rang + bar_width, data_dict['attributeNames'][-3:], rotation=45)
    xlabel('Attributes')
    ylabel('Component coefficients values')
    legend(legend_stings)
    grid()
    title('PCA Components Coefficients')
    show()

def attributes_histogram(data_dict):
    figure()
    rows = np.floor(np.sqrt(len(data_dict['attributeNames'][0:8])))
    columns = np.ceil(float(len(data_dict['attributeNames'][0:8]))/rows)
    for i in range(len(data_dict['attributeNames'][0:8])):
        subplot(int(rows),int(columns),i+1)
        hist(data_dict['X'][:,i],log=True)
        #xlabel(str(i))
        #xlabel(data_dict['attributeNames'][i])
        if i==0: title('Historgram of 8 word frequency attributes') 
    plt.subplots_adjust(wspace=0.5)
    show()

    #show last 3 attributes
    figure()
    rows=1
    columns=3
    for i in range(3):
        subplot(int(rows),int(columns),i+1)
        hist(data_dict['X'][:,-4+i:-3+i],log=True)
        xlabel(data_dict['attributeNames'][-3+i], fontsize=8)
        if i == 0: title('Histogram of last 3 Attribute')
    plt.subplots_adjust(wspace=1)
    show()
