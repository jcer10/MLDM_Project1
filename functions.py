import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend,boxplot,bar,xticks,grid,subplot,hist
from matplotlib.pylab import semilogx
import seaborn as sns
from toolbox_02450 import rlr_validate, correlated_ttest, train_neural_net, draw_neural_net

import torch
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from matplotlib.pylab import loglog
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


def regression_part_a(data_dict):
    #The attribute (word_freq_dirct) that we are going to use as 'y' for the regression
    pred_index=39

    X=(data_dict["X"] - np.ones((data_dict["N"], 1)) * data_dict["X"].mean(axis=0)) / data_dict["X"].std(axis=0)
    y=X[:,pred_index]
    #remove the attribute from X
    X=np.delete(X,pred_index,axis=1)

    attributeNames=data_dict["attributeNames"]
    attributeNames=np.delete(attributeNames,pred_index,axis=0)

    #create a list of lambdas
    lambdas = np.power(2.,range(-10,30))


    N_attributes = X.shape[1]

    #weights
    weights_rlr = np.empty((N_attributes, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, shuffle=True)

    validation_folds = 10
    print("Using cross-validation with ", validation_folds," folds")
    val_error_opt, opt_lambda, mean_w_vs_lambda, train_error_lambda, test_error_lambda = rlr_validate(X_train,
                                                                                                      y_train,
                                                                                                      lambdas,
                                                                                                      validation_folds)

    Xtrans_y = X_train.T @ y_train
    Xtrans_X = X_train.T @ X_train

    # Estimate weights of attributes considering the optimal lambda
    lambda_I = opt_lambda * np.eye(N_attributes)
    lambda_I[0, 0] = 0  # Do no regularize the bias term
    weights_rlr[:, 0] = np.linalg.solve(Xtrans_X + lambda_I, Xtrans_y).squeeze()

    # MSE with optimal lambda
    Training_error_rlr = np.square(y_train - X_train @ weights_rlr[:, 0]).sum(axis=0) / y_train.shape[0]
    Test_error_rlr = np.square(y_test - X_test @ weights_rlr[:, 0]).sum(axis=0) / y_test.shape[0]

    print("Best lambda: ", opt_lambda, " | Error: ", Test_error_rlr)

    print('Linear weights:')
    for m in range(N_attributes):
        print(attributeNames[m], np.round(weights_rlr[m, 0], ))

    # Display the results
    figure()
    title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
    loglog(lambdas, train_error_lambda.T, 'b.-', lambdas, test_error_lambda.T, 'r.-')
    xlabel('Regularization factor')
    ylabel('Squared error (crossvalidation)')
    legend(['Train error', 'Validation error'])
    grid()
    show()


def ann_inner_folds(X, y, h_range, cvf = 10):
    CV = model_selection.KFold(cvf, shuffle=True)
    N_attributes = X.shape[1]
    w = np.empty((N_attributes, cvf, len(h_range)))
    train_error = np.empty((cvf, len(h_range)))
    test_error = np.empty((cvf, len(h_range)))
    f = 0
    for train_index, test_index in CV.split(X, y):
        X_train = torch.Tensor(X[train_index])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index])
        y_test = torch.Tensor(y[test_index])

        for ind in range(0, len(h_range)):
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(N_attributes, h_range[ind]),
                torch.nn.Tanh(),  # activation function,
                torch.nn.Linear(h_range[ind], 1))
            loss = torch.nn.MSELoss()
            net, final_loss, learning_curve = train_neural_net(model,loss,X=X_train,y=y_train,n_replicates=1,max_iter=10000)

            #class labes
            y_train_est = net(X_train).detach().numpy()
            y_test_est = net(X_test).detach().numpy()

            train_error[f, ind] = np.square(y_train - y_train_est).sum(axis=0) / y_train.shape[0]
            test_error[f, ind] = np.square(y_test - y_test_est).sum(axis=0) / y_test.shape[0]

        f = f + 1

    optimal_h = h_range[np.argmin(np.mean(test_error, axis=0))]

    return optimal_h


def regression_part_b(data_dict):

    #Process data
    pred_index = 39

    X = (data_dict["X"] - np.ones((data_dict["N"], 1)) * data_dict["X"].mean(axis=0)) / data_dict["X"].std(axis=0)
    y = X[:, pred_index]
    X = np.delete(X, pred_index, axis=1)

    N_attributes = X.shape[1]

    attributeNames = data_dict["attributeNames"]
    attributeNames = np.delete(attributeNames, pred_index, axis=0)

    # Create crossvalidation
    K_outer_fold = 5
    K_inner_fold = 5
    print()
    print("Using K_outer & K_inner: ", K_outer_fold, " & ", K_inner_fold)
    print()
    CV = model_selection.KFold(K_outer_fold, shuffle=True)


    lambdas = np.power(10., range(0,11))

    weights_rlr = np.empty((N_attributes, K_outer_fold))


    #hyperparameters values for neural network
    h_values = [1, 5, 10, 15, 20, 50]

    # Initialize empty arrays
    Train_error_rlr = np.empty((K_outer_fold, 1))
    Test_error_rlr = np.empty((K_outer_fold, 1))
    Train_error_baseline = np.empty((K_outer_fold, 1))
    Test_error_baseline = np.empty((K_outer_fold, 1))
    Train_error_ann = np.empty((K_outer_fold, 1))
    Test_error_ann = np.empty((K_outer_fold, 1))
    cross_lambdas = []
    cross_h = []

    print("Y length: ", len(y))
    print()
    figure()
    plot(y)
    show()

    k = 0
    for train_index, test_index in CV.split(X, y):
        if k == 0:
            print("Training length: ", len(train_index))
            print("Test length: ", len(test_index))
            print()

        print("K=", k)

        # Optimal lambda for regression
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        val_eror_opt, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train,
                                                                                                          y_train,
                                                                                                          lambdas,
                                                                                                          K_inner_fold)
        print("Best lambda: ", opt_lambda)
        cross_lambdas.append(opt_lambda)
        Xtrans_y = X_train.T @ y_train
        Xtrans_X = X_train.T @ X_train

        # Estimate weights
        lambda_I = opt_lambda * np.eye(N_attributes)
        lambda_I[0, 0] = 0  # Do no regularize the bias term
        weights_rlr[:, k] = np.linalg.solve(Xtrans_X + lambda_I, Xtrans_y).squeeze()

        # MSE
        Train_error_rlr[k] = np.square(y_train - X_train @ weights_rlr[:, k]).sum(axis=0) / y_train.shape[0]
        Test_error_rlr[k] = np.square(y_test - X_test @ weights_rlr[:, k]).sum(axis=0) / y_test.shape[0]
        print("Error linear: ", Test_error_rlr[k])

        if k == K_outer_fold - 1:
            figure(1)
            plot(y_train, c='black')
            plot(X_train @ weights_rlr[:, k], c='red')
            figure(2)
            plot(y_test, c='black')
            plot(X_test @ weights_rlr[:, k], c='red')
            show()
            #Lambdas
            figure(k, figsize=(12, 8))
            subplot(1, 2, 1)
            semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], '.-')
            xlabel('Regularization factor')
            ylabel('Mean Coefficient Values')
            grid()
            subplot(1, 2, 2)
            title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
            loglog(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
            xlabel('Regularization factor')
            ylabel('Squared error (crossvalidation)')
            legend(['Train error', 'Validation error'])
            grid()
            show()

        #Baseline model


        Train_error_baseline[k] = np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
        Test_error_baseline[k] = np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
        print("Error baseline: ", Test_error_baseline[k])

        if k == K_outer_fold - 1:
            figure(1)
            plot(y_train, c='black')
            plot([y_train.mean()] * y_train.shape[0], c='red')
            figure(2)
            plot(y_test, c='black')
            plot([y_test.mean()] * y_test.shape[0], c='red')
            show()

        #Neural Network


        X_train = torch.Tensor(X[train_index])
        y_train = torch.Tensor(np.reshape(y[train_index],(len(train_index),1)))
        X_test = torch.Tensor(X[test_index])
        y_test = torch.Tensor(np.reshape(y[test_index],(len(test_index),1)))


        n_hidden_units = ann_inner_folds(X_train, y_train, h_values, K_inner_fold)
        print("Best h: ", n_hidden_units)
        cross_h.append(n_hidden_units)
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(N_attributes, n_hidden_units),
            torch.nn.Tanh(),  # activation function
            torch.nn.Linear(n_hidden_units, 1))
        loss = torch.nn.MSELoss()
        net, final_loss, learning_curve = train_neural_net(model,loss,X=X_train,y=y_train,n_replicates=1,max_iter=10000)

        # estimated class labels
        y_train_est = net(X_train).detach().numpy()
        y_test_est = net(X_test).detach().numpy()
        Train_error_ann[k] = np.square(y_train - y_train_est).sum(axis=0) / y_train.shape[0]
        Test_error_ann[k] = np.square(y_test - y_test_est).sum(axis=0) / y_test.shape[0]
        print("Error ann: ", Test_error_ann[k])

        if k == K_outer_fold - 1:
            figure(1)
            plot(y_train, c='black')
            plot(y_train_est, c='red')
            figure(2)
            plot(y_test, c='black')
            plot(y_test_est, c='red')
            show()

            weights = [net[i].weight.data.numpy().T for i in [0, 2]]
            biases = [net[i].bias.data.numpy() for i in [0, 2]]
            tf = [str(net[i]) for i in [0, 2]]
            draw_neural_net(weights, biases, tf)

        k += 1

    print("Linear train error mean: ", Train_error_rlr.mean())
    print("Linear test error mean: ", Test_error_rlr.mean())
    print('--------------------------------------------------------------------------')
    print("Baseline train error mean: ", Train_error_baseline.mean())
    print("Baseline test error mean: ", Test_error_baseline.mean())
    print('--------------------------------------------------------------------------')
    print("Ann train error mean: ", Train_error_ann.mean())
    print("Ann test error mean: ", Test_error_ann.mean())
    print('--------------------------------------------------------------------------')
    print("Best lambda: ", cross_lambdas[np.argmin(Test_error_rlr)], " | Error: ",
          Test_error_rlr[np.argmin(Test_error_rlr)])
    print("Best h: ", cross_h[np.argmin(Test_error_ann)], " | Error: ", Test_error_ann[np.argmin(Test_error_ann)])

    #Linear weights
    print('Regression weights in the last cross validation fold:')
    for m in range(N_attributes):
        print(attributeNames[m], np.round(weights_rlr[m, -1], 2))


    # Evaluation
    K_outer_fold = 5
    K_inner_fold = 5

    CV = model_selection.KFold(K_outer_fold, shuffle=True)


    optimal_lambda = cross_lambdas[np.argmin(Test_error_rlr)]
    weights_rlr = np.empty((N_attributes, K_outer_fold))

    optimal_h = cross_h[np.argmin(Test_error_ann)]

    # Initialize variables
    y_rlr_real = []
    y_rlr_pred = []
    y_baseline_real = []
    y_baseline_pred = []
    y_ann_real = []
    y_ann_pred = []

    print("Using optimal lambda: ", optimal_lambda)
    print("Using optimal h: ", optimal_h)

    k = 0
    for train_index, test_index in CV.split(X, y):
        print("K=", k)

        #Regression
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        Xtrans_y = X_train.T @ y_train
        Xtrans_X = X_train.T @ X_train

        lambda_I = optimal_lambda * np.eye(N_attributes)
        lambda_I[0, 0] = 0
        weights_rlr[:, k] = np.linalg.solve(Xtrans_X + lambda_I, Xtrans_y).squeeze()

        y_rlr_real = y_rlr_real + list(y_test)
        y_rlr_pred = y_rlr_pred + list(X_test @ weights_rlr[:, k])

        #Baseline

        y_baseline_real = y_baseline_real + list(y_test)
        y_baseline_pred = y_baseline_pred + [y_test.mean()] * len(y_test)

        #Neural Netwrok
        X_train = torch.Tensor(X[train_index])
        y_train = torch.Tensor(np.reshape(y[train_index],(len(train_index),1)))
        X_test = torch.Tensor(X[test_index])
        y_test = torch.Tensor(np.reshape(y[test_index],(len(test_index),1)))

        n_hidden_units = optimal_h
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(N_attributes, n_hidden_units),
            torch.nn.Tanh(),  # activation function
            torch.nn.Linear(n_hidden_units, 1), )
        loss = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss
        net, final_loss, learning_curve = train_neural_net(model,loss,X=X_train,y=y_train,n_replicates=1,max_iter=10000)

        y_train_est = net(X_train).detach().numpy()
        y_test_est = net(X_test).detach().numpy()

        y_network_real = y_ann_real + list(y_test.numpy()[:, 0])
        y_network_pred = y_ann_pred + list(y_test_est[:, 0])

        if k == K_outer_fold - 1:

            weights = [net[i].weight.data.numpy().T for i in [0, 2]]
            biases = [net[i].bias.data.numpy() for i in [0, 2]]
            tf = [str(net[i]) for i in [0, 2]]
            draw_neural_net(weights, biases, tf)

        k += 1

    print()
    print('linear weights in last fold:')
    for m in range(N_attributes):
        print(attributeNames[m], np.round(weights_rlr[m, -1], 2))
    print()

    error_rlr_square = np.square(np.array(y_rlr_real) - np.array(y_rlr_pred))
    error_baseline_square = np.square(np.array(y_baseline_real) - np.array(y_baseline_pred))
    error_network_square = np.square(np.array(y_network_real) - np.array(y_network_pred))
    print("Linear error mean: ", error_rlr_square.mean())
    print("Baseline error mean: ", error_baseline_square.mean())
    print("Ann error mean: ", error_network_square.mean())
    print()

    error_rlr_vs_baseline = error_rlr_square - error_baseline_square
    error_rlr_vs_network = error_rlr_square - error_network_square
    error_network_vs_baseline = error_network_square - error_baseline_square
    print("Error linear vs baseline: ", error_rlr_vs_baseline.mean())
    print("Error linear vs ann: ", error_rlr_vs_network.mean())
    print("Error ann vs baseline: ", error_network_vs_baseline.mean())
    print()

    # Initialize parameters and run test appropriate for setup II
    alpha = 0.05
    rho = 1 / K_outer_fold
    p_setupII, CI_setupII = correlated_ttest(error_rlr_vs_baseline, rho, alpha=alpha)
    p2_setupII, CI2_setupII = correlated_ttest(error_rlr_vs_network, rho, alpha=alpha)
    p3_setupII, CI3_setupII = correlated_ttest(error_network_vs_baseline, rho, alpha=alpha)
    print("p (linear vs baseline) : ", p_setupII)
    print("CI (linear vs baseline): ", CI_setupII)
    print("p (linear vs ann): ", p2_setupII)
    print("CI (linear vs ann): ", CI2_setupII)
    print("p (ann vs baseline): ", p3_setupII)
    print("CI (ann vs baseline): ", CI3_setupII)

    return