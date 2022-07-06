import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.ndimage import shift


# compute LMS algorithm for N iterations  
# w[k+1] = w[k] + 2*alpha*x[k]e[k]
def LMS(x_stack, y, alpha, max_iter, w):
    """
    :param np.arryr x_stack : x[k]
    :param np.array y : reference data_y
    :param float alpha: learning rate
    :param int max_iter: number of iterations
    :param np.array w : weight vector
    """

    # initialize weights history
    w_history = np.zeros((max_iter+1, 3))

    # initialize weights
    w_history[0, :] = w.T

    # compute lms
    for i in range(max_iter):
        x_k = x_stack[[i]].T
        error = y[i]-np.dot(w.T, x_k)
        #update weights
        w = w + 2*alpha*error.item()*x_k
        #store weight
        w_history[[i+1], :] = w.T   

    return w_history

# compute NLMS algorithm for N iterations  
# w[k+1] = w[k] + 2*alpha/ (sigma) *x[k]e[k]
def NLMS(x_stack, y, alpha, max_iter, w):
    """
    :param np.arryr x_stack : x[k]
    :param np.array y : reference data_y
    :param float alpha: learning rate
    :param int max_iter: number of iterations
    :param np.array w : weight vector
    """

    # initialize weights history
    w_history = np.zeros((max_iter+1, 3))

    # initialize weights
    w_history[0, :] = w.T

    # compute nlms
    for i in range(max_iter):
        x_k = x_stack[[i]].T
        error = y[i]-np.dot(w.T, x_k)
        sigma = (x_k.T@x_k)/3 + 0.01
        #update weights
        w = w + 2*(alpha/sigma)*error.item()*x_k
        #store weight
        w_history[[i+1], :] = w.T   

    return w_history

# compute RLS algorithm for N iterations  
# w[k+1] = np.dot(inv(R_x_hat), r_yx_hat)
def RLS(x_stack, y, gamma, delta_inv, max_iter, w_init):
    """
    :param np.arryr x_stack : x[k]
    :param np.array y : reference data_y
    :param float gamma: forgetting factor
    :param float delta_inv: inverse of the auto-correlation matrix of x[k]
    :param int max_iter: number of iterations
    :param np.array w_init : weight vector
    """
    # initialize weights history
    w_history = np.zeros((max_iter, 3))

    # initialize weights
    w_history[0, :] = w_init.T

    # initialize auto-correlation matrix
    R_x_inv = delta_inv * np.identity(3)
    r_yx = np.zeros((3, 1))

    # compute parameters
    for i in range(max_iter-1):

        # input/output data    
        x_k = x_stack[[i]].T
        y_k = y[i]

        # calculate parameters
        g = np.dot(R_x_inv, x_k) / (gamma**2 + np.dot(np.dot(x_k.T, R_x_inv), x_k))
        R_x_inv = gamma**(-2) * (R_x_inv - np.dot(np.dot(g, x_k.T), R_x_inv))
        r_yx = (gamma**2)*r_yx + x_k*y_k

        # update weights
        w = R_x_inv@r_yx        

        # store weights
        w_history[[i+1], :] = w.T

    return w_history

    
# contour plot function
def contour_plot(w0, w1, w_train, J_vals, title, filename, show=False):

    # plot the contour plot
    fig = plt.figure()
    cp = plt.contour(w0, w1, J_vals)
    plt.clabel(cp, inline=1, fontsize=10)
    plt.plot(w_train[:,0], w_train[:,1])
    plt.title(title)
    plt.xlabel('w0')
    plt.ylabel('w1')

    # plot optimal weights as point in contourplot
    plt.plot(0.2, 1, 'x', color='red', markersize=10)
    
    if show:
        plt.show()

    # save figure as png
    figure_name_png = f"figures/{filename}.png"
    figure_name_eps = f"figures/{filename}.eps"
    fig.savefig(figure_name_eps)
    fig.savefig(figure_name_png, dpi=300)



if __name__ == "__main__":
    print(" ---- start ---- ")

    # Load the data
    data = pd.read_csv("assignment1_data.csv", header=None)

    # split data into x and y
    X, y = np.array(data.iloc[:, 0]), np.array(data.iloc[:,1])
    x_stack = np.vstack((X, shift(X, 1), shift(X, 2))).T


    # initialize the filter
    w_2_const = -0.5 # for countourplots    
    N = data.shape[0]
    w_init = np.zeros((3, 1))

    # set parameters for each algorithm
    alpha = 0.0005
    gamma = 1 - 1e-4
    delta_inv = 0.001
    
        
    """    
    # apply mean square error
    w_lms = Least_Mean_Square(x_stack, data_y, alpha, N, w)
    # apply normalized mean square error
    w_nlms = Normlized_LMS(x_stack, data_y, alpha, N, w) 
    """

    w_RLS = RLS(x_stack, y, gamma, delta_inv, N, w_init)       # apply RLS
    print(r"Optimal w_{RLS}=", w_RLS[-1])
    w_LMS = LMS(x_stack, y, alpha, N, w_init)       # apply LMS
    print(r"Optimal w_{LMS}=", w_LMS[-1])
    w_NLMS = NLMS(x_stack, y, alpha, N, w_init)       # apply NLMS
    print(r"Optimal w_{NLMS}=", w_NLMS[-1])


    # Plot the trajectory of the filter coefficients as they evolve, together with a contour
    # plot of the objective function J. 
    w0 = np.linspace(-0.5, 0.5, 100)
    w1 = np.linspace(-0.1, 1.5, 100)
    W0, W1 = np.meshgrid(w0, w1)
    J_vals = np.zeros((len(W0), len(W1)))

    # compute the objective function J for each point in the grid
    for i in range(len(W0)):
        for j in range(len(W1)):
            w_temp = np.array([W0[0, i], W1[j, 0], w_2_const]).T
            J_vals[i, j] =  mean_squared_error(y,x_stack @ w_temp)


    # plot the contour plot for LMS
    contour_plot(W0, W1, w_LMS, J_vals, 
        title=rf"LMS Contour Plot for $\alpha$ = {alpha}", 
        filename=f"LMS_contour_plot_alpha_{alpha}")
    
    # plot the contour plot for NLMS
    contour_plot(W0, W1, w_NLMS, J_vals,
        title=rf"NLMS Contour Plot for $\alpha$ = {alpha}",
        filename=f"NLMS_contour_plot_alpha_{alpha}")

    # plot the contour plot for RLS
    contour_plot(W0, W1, w_RLS, J_vals,
        title=rf"RLS Contour Plot for $\gamma$ = {gamma}",
        filename=f"RLS_contour_plot_gamma_{gamma}")