import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss

def hist_resids(y_test, y_score, model):
    """
    Plots the residuals as histogram
    """
    ## first compute vector of residuals. 
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    fig = plt.figure(figsize=(16,6))
    sns.distplot(resids)
    plt.title('Histogram of residuals ({})'.format(model))
    plt.xlabel('Residual value')
    plt.ylabel('count')
    plt.show()

def resid_qq(y_test, y_score, model):
    """
    Plots the offset between test and prediction values
    """
    ## first compute vector of residuals. 
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    fig = plt.figure(figsize=(16,6))
    ss.probplot(resids.flatten(), plot = plt)
    plt.title('Residuals vs. predicted values ({})'.format(model))
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')
    plt.show()

def resid_plot(y_test, y_score, model):
    """
    Plots the residuals
    """
    ## first compute vector of residuals. 
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    fig = plt.figure(figsize=(16,6))
    sns.regplot(y_score, resids, fit_reg=False)
    plt.title('Residuals vs. predicted values ({})'.format(model))
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')    

def show_pred_vs_test(y_test, y_score, model):
    """
    Compares predicted to test values
    """
    samples = 60
    fig = plt.figure(figsize=(16,6))
    x = range(samples)
    y1 = y_test[:samples].reshape(-1)
    y2 = y_score[:samples].reshape(-1)
    dy = (y2-y1)/2
    ye = (y1+y2)/2
    plt.scatter(x, y1, c='red')
    plt.scatter(x, y2, c='blue')
    plt.errorbar(x, ye, yerr=dy, fmt='.');
    plt.ylabel("Rate spread")
    plt.xlabel("Test sample")
    plt.title("Predicted vs. test value ({})".format(model))
    plt.show()    

def print_metrics(y_true, y_predicted, n_parameters):
    """
    Prints the model's performance to the console
    """
    ## First compute R^2 and the adjusted R^2
    r2 = sklm.r2_score(y_true, y_predicted)
    r2_adj = r2 - (n_parameters - 1)/(y_true.shape[0] - n_parameters) * (1 - r2)    
    ## Print the usual metrics and the R^2 values
    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))
    print('Adjusted R^2           = ' + str(r2_adj))
