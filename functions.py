import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend,boxplot,bar,xticks,grid,subplot,hist
from matplotlib.pylab import semilogx
from toolbox_02450 import rlr_validate, correlated_ttest, train_neural_net, draw_neural_net, mcnemar

import torch
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib.pylab import loglog

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
    weights_rlr = np.zeros((N_attributes, 1))

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
        print(attributeNames[m], weights_rlr[m, 0])

    # Display the results
    figure()
    title('Optimal lambda: '+str(opt_lambda))
    loglog(lambdas, train_error_lambda.T, 'b.-', lambdas, test_error_lambda.T, 'r.-')
    xlabel('Regularization factor (lambda)')
    ylabel('Squared error')
    legend(['Train error', 'Validation error'])
    grid()
    show()


def ann_inner_folds(X, y, h_list, cv_folds = 5):
    CV = model_selection.KFold(cv_folds, shuffle=True)
    N_attributes = X.shape[1]
    w = np.zeros((N_attributes, cv_folds, len(h_list)))
    train_error = np.zeros((cv_folds, len(h_list)))
    test_error = np.zeros((cv_folds, len(h_list)))
    fold = 0
    for train_index, test_index in CV.split(X, y):
        X_train = torch.Tensor(X[train_index])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index])
        y_test = torch.Tensor(y[test_index])

        for ind in range(0, len(h_list)):
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(N_attributes, h_list[ind]),
                torch.nn.Tanh(),  # activation function,
                torch.nn.Linear(h_list[ind], 1))
            loss = torch.nn.MSELoss()
            net, final_loss, learning_curve = train_neural_net(model,loss,X=X_train,y=y_train,n_replicates=1,max_iter=10000)

            #class labes
            y_train_est = net(X_train).detach().numpy()
            y_test_est = net(X_test).detach().numpy()

            train_error[fold, ind] = np.square(y_train - y_train_est).sum(axis=0) / y_train.shape[0]
            test_error[fold, ind] = np.square(y_test - y_test_est).sum(axis=0) / y_test.shape[0]

        fold = fold + 1

    optimal_h = h_list[np.argmin(np.mean(test_error, axis=0))]

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

    CV = model_selection.KFold(K_outer_fold, shuffle=True)



    lambdas = np.power(10., range(0,11))
    weights_rlr = np.zeros((N_attributes, K_outer_fold))


    # Initialize arrays
    Train_error_rlr = np.zeros((K_outer_fold, 1))
    Test_error_rlr = np.zeros((K_outer_fold, 1))
    Train_error_baseline = np.zeros((K_outer_fold, 1))
    Test_error_baseline = np.zeros((K_outer_fold, 1))
    Train_error_ann = np.zeros((K_outer_fold, 1))
    Test_error_ann = np.zeros((K_outer_fold, 1))
    cross_lambdas = []
    cross_h = []

    # hyperparameters values for neural network
    h_values = [1, 10, 20, 50, 100]




    k = 0
    for train_index, test_index in CV.split(X, y):
        print('--------------------------------------------------------------------------')
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
        print("Optimal lambda: ", opt_lambda)
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
        print("Error linear regression: ", Test_error_rlr[k])

        if k == K_outer_fold-1:
            figure(1)
            title("Linear Regression, train predictions")
            plot(y_train, c='black')
            plot(X_train @ weights_rlr[:, k], c='red')
            figure(2)
            title("Linear Regression, test predictins")
            plot(y_test, c='black')
            plot(X_test @ weights_rlr[:, k], c='red')
            show()
            #Lambdas
            figure(k)
            subplot(1, 2, 1)
            semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], '.-')
            xlabel('Regularization factor (lambda)')
            ylabel('Mean Coeff Values')
            grid()
            subplot(1, 2, 2)
            title('Optimal lambda: ' +str(opt_lambda))
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

        if k == K_outer_fold :
            figure(1)
            title("Baseline model, train predictions")
            plot(y_train, c='black')
            plot([y_train.mean()] * y_train.shape[0], c='red')
            figure(2)
            title("Baseline model, test predictions")
            plot(y_test, c='black')
            plot([y_test.mean()] * y_test.shape[0], c='red')
            show()

        #Neural Network

        X_train = torch.Tensor(X[train_index])
        y_train = torch.Tensor(np.reshape(y[train_index],(len(train_index),1)))
        X_test = torch.Tensor(X[test_index])
        y_test = torch.Tensor(np.reshape(y[test_index],(len(test_index),1)))


        units = ann_inner_folds(X_train, y_train, h_values, K_inner_fold)

        print("Best h: ", units)
        cross_h.append(units)
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(N_attributes, units),
            torch.nn.Tanh(),  # activation function
            torch.nn.Linear(units, 1))
        loss = torch.nn.MSELoss()
        net, final_loss, learning_curve = train_neural_net(model,loss,X=X_train,y=y_train,n_replicates=1,max_iter=10000)

        # estimated class labels
        y_train_est = net(X_train).detach().numpy()
        y_test_est = net(X_test).detach().numpy()
        Train_error_ann[k] = np.square(y_train - y_train_est).sum(axis=0) / y_train.shape[0]
        Test_error_ann[k] = np.square(y_test - y_test_est).sum(axis=0) / y_test.shape[0]
        print("Error neural network: ", Test_error_ann[k])

        if k == K_outer_fold-1:
            figure(1)
            title("Neural Network, train predictions")
            plot(y_train, c='black')
            plot(y_train_est, c='red')
            figure(2)
            title("Neural Network, test predictions")
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
    print('--------------------------------------------------------------------------')
    print("Evaluation")
    K_outer_fold = 5

    CV = model_selection.KFold(K_outer_fold, shuffle=True)


    optimal_lambda = cross_lambdas[np.argmin(Test_error_rlr)]
    weights_rlr = np.zeros((N_attributes, K_outer_fold))

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

        units = optimal_h
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(N_attributes, units),
            torch.nn.Tanh(),  # activation function
            torch.nn.Linear(units, 1), )
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

    print('Regression weights in last fold of evaluation:')
    for m in range(N_attributes):
        print(attributeNames[m], np.round(weights_rlr[m, -1], 2))

    error_rlr_square = np.square(np.array(y_rlr_real) - np.array(y_rlr_pred))
    error_baseline_square = np.square(np.array(y_baseline_real) - np.array(y_baseline_pred))
    error_network_square = np.square(np.array(y_network_real) - np.array(y_network_pred))
    print("Linear error mean: ", error_rlr_square.mean())
    print("Baseline error mean: ", error_baseline_square.mean())
    print("Neural network error mean: ", error_network_square.mean())

    error_rlr_vs_baseline = error_rlr_square - error_baseline_square
    error_rlr_vs_network = error_rlr_square - error_network_square
    error_network_vs_baseline = error_network_square - error_baseline_square
    print("Error linear vs baseline: ", error_rlr_vs_baseline.mean())
    print("Error linear vs neural network: ", error_rlr_vs_network.mean())
    print("Error neural network vs baseline: ", error_network_vs_baseline.mean())

    # test II
    alpha = 0.05
    rho = 1 / K_outer_fold
    p1,ci1 = correlated_ttest(error_rlr_vs_baseline, rho, alpha=alpha)
    p2, ci2 = correlated_ttest(error_rlr_vs_network, rho, alpha=alpha)
    p3, ci3= correlated_ttest(error_network_vs_baseline, rho, alpha=alpha)
    print("p-value  Linear regression vs Baseline: ", p1)
    print("Confidence Interval Linear Regression vs Baseline: ", ci1)
    print("p-value Linear Regression vs Neural Network: ", p2)
    print("Confidence Interval Linear regression vs Neural Network: ", ci2)
    print("p-value Neural Network vs Baseline: ", p3)
    print("Confidence Interval Neural Netowrk vs Baseline: ", ci3)

    return

def classification_part(data_dict):

    # Processing data
    X = (data_dict["X"] - np.ones((data_dict["N"], 1)) * data_dict["X"].mean(axis=0)) / data_dict["X"].std(axis=0)
    y = data_dict['y']
    attributeNames = data_dict['attributeNames']
    classNames = data_dict['classNames']
    N, M = X.shape
    C = len(classNames)

    K_outer = 5
    K_inner = 5
    CV = model_selection.KFold(K_outer,shuffle=True)

    # Logistic Regression
    lambdas = np.power(10., range(-3,3))
    lr_models = []
    E_lr_outer = np.zeros(K_outer)
    y_lr_pred = []

    # ANN
    h_vals = np.arange(15,20)
    ann_models =  []
    E_ann_outer = np.zeros(K_outer)
    y_ann_pred = []

    # Baseline
    E_bl_outer  = np.zeros(K_outer)
    y_bl_pred = []

    y_true = []

    # Table for results
    results = pd.DataFrame(columns=['i', 'h', 'E_ann_outer', 'lambda_i', 'E_lr_outer', 'E_bl_outer'])

    # Optimal parameters for each outer fold
    best_lambda = np.zeros(K_outer)
    best_h = np.zeros(K_outer)

    # Parameters value for ANN models
    loss_fn = torch.nn.BCELoss()
    max_iter = 10000

    
    # Create the Logistic Regression models
    for i in range(len(lambdas)):
        mdl = LogisticRegression(penalty='l2', C=1/lambdas[i], max_iter = max_iter)
        lr_models.append(mdl)

    # Create the Neural Networks models for binary classification
    for i in range(len(h_vals)):
        model_ann = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, h_vals[i]),
                            torch.nn.Tanh(),
                            torch.nn.Linear(h_vals[i], 1),
                            torch.nn.Sigmoid()
                            )
        ann_models.append(model_ann)


    # Two-layer cross-validation
    for k, (train_outer_index, test_outer_index) in enumerate(CV.split(X,y)): 
        print('\nCrossvalidation outer-fold: {0}/{1}'.format(k+1,K_outer))    

        X_train = X[train_outer_index,:]
        y_train = y[train_outer_index]
        X_test  = X[test_outer_index,:]
        y_test  = y[test_outer_index]
        
        E_lr_inner = np.zeros((K_inner, len(lambdas)))
        E_ann_inner = np.zeros((K_inner, len(h_vals)))


        for j, (train_inner_index, test_inner_index) in enumerate(CV.split(X_train,y_train)): 
            print("===============================================================")
            print('\nCrossvalidation inner-fold: {0}/{1}'.format(j+1,K_inner)) 

            X_train_inner = X_train[train_inner_index,:]
            y_train_inner = y_train[train_inner_index]
            X_test_inner  = X_train[test_inner_index,:]
            y_test_inner  = y_train[test_inner_index]


            # Train the models (Inner Loop)
            # --------------------------------

            # Logistic Regression'
            for i in range(0, len(lambdas)):
                lr_models[i].fit(X_train_inner, y_train_inner)
                y_test_inner_est = lr_models[i].predict(X_test_inner).T
                test_error_rate_inner = np.sum(y_test_inner_est != y_test_inner) / len(y_test_inner)
                E_lr_inner[j, i] = test_error_rate_inner


            # ANN
            for i in range(len(h_vals)):
                net, _, _ = train_neural_net(ann_models[i],
                                                        loss_fn,
                                                        X=torch.Tensor(X_train_inner),
                                                        y=torch.tensor(y_train_inner, dtype=torch.float).unsqueeze(1),
                                                        n_replicates=1,
                                                        max_iter=max_iter)
                
                y_sigmoid = net(torch.tensor(X_test_inner, dtype=torch.float))
                y_test_est_inner = (y_sigmoid>.5).type(dtype=torch.uint8).squeeze().data.numpy()
                e_inner = (y_test_est_inner != y_test_inner)
                E_ann_inner[j,i] = sum(e_inner) / len(y_test_inner)
            

            print("Logistic Regression Inner Loop Error Rate: ", E_lr_inner[j,:])
            print("ANN Inner Loop Error Rate: ", E_ann_inner[j,:])
            print("===============================================================")



        # Optimal lambda
        mean_error = np.mean(E_lr_inner, axis = 0)
        opt_lambda_idx = np.argmin(mean_error)
        
        # Optimal h value
        mean_error = np.mean(E_ann_inner, axis = 0)
        opt_hval_idx = np.argmin(mean_error)



        # Testing (outer loop)
        # --------------------------------


        # Logistic Regression
        lr_models[opt_lambda_idx].fit(X_train, y_train)
        
        y_test_est = lr_models[opt_lambda_idx].predict(X_test).T
        y_lr_pred.append(y_test_est)
        test_error_rate = np.sum(y_test_est != y_test) / len(y_test)
        E_lr_outer[k] = test_error_rate

        best_lambda[k] = lambdas[opt_lambda_idx]



        # ANN
        net, final_loss, learning_curve = train_neural_net(ann_models[opt_hval_idx],
                                                        loss_fn,
                                                        X = torch.tensor(X_train, dtype=torch.float),
                                                        y = torch.tensor(y_train, dtype=torch.float).unsqueeze(1),
                                                        n_replicates=1,
                                                        max_iter=max_iter)
        
        y_sigmoid = net(torch.tensor(X_test, dtype=torch.float))
        y_test_est = (y_sigmoid>.5).type(dtype=torch.uint8).squeeze().data.numpy()
        y_ann_pred.append(y_test_est)
        e = (y_test_est != y_test)
        E_ann_outer[k] = sum(e) / len(y_test)

        best_h[k] = h_vals[opt_hval_idx]


        # Baseline
        y_test_est = np.ones(y_test.shape)*np.bincount(y_train.astype(int)).argmax()
        y_bl_pred.append(y_test_est)
        e = (y_test_est != y_test)
        E_bl_outer[k] = sum(e) / len(y_test)

        y_true.append(y_test)
            
        #results.append([k+1, best_h[k], E_ann_outer[k], best_lambda[k], E_lr_outer[k], E_bl_outer[k]])
        results.loc[len(results)] = [k+1, best_h[k], E_ann_outer[k], best_lambda[k], E_lr_outer[k], E_bl_outer[k]]

    
    print(results)

    y_true = np.concatenate(y_true)
    y_lr_pred = np.concatenate(y_lr_pred)
    y_ann_pred = np.concatenate(y_ann_pred)
    y_bl_pred = np.concatenate(y_bl_pred)


    # Comparison of the models

    # lr vs ann
    [thetahat, CI, p] = mcnemar(y_true, y_lr_pred, y_ann_pred)
    print('lr vs ann')
    print('theta: {0}'.format(thetahat), 'CI: {0}'.format(CI), 'p-value: {0}'.format(p))


    # lr vs bl
    [thetahat2, CI2, p2] = mcnemar(y_true, y_lr_pred, y_bl_pred)
    print('lr vs bl')
    print('theta:     {0}'.format(thetahat2), 'CI: {0}'.format(CI2), 'p-value: {0}'.format(p2))

    # ann vs bl
    [thetahat3, CI3, p3] = mcnemar(y_true, y_ann_pred, y_bl_pred)

    print('ann vs bl')
    print('theta:     {0}'.format(thetahat3), 'CI: {0}'.format(CI3), 'p-value: {0}'.format(p3))


    # Final Logistic regression model

    lambda_val = 0.1

    lr_model = LogisticRegression(penalty='l2', C=1/lambda_val, max_iter = max_iter)
    lr_model.fit(X, y)
    y_train_est_final = lr_model.predict(X).T

    w_est = lr_model.coef_[0]
    w_est = [round(w, 4) for w in w_est]
    
    weights = pd.DataFrame(columns = attributeNames)
    weights.loc[len(weights)] = w_est
    weights = weights.T

    print(weights.to_latex())

    return