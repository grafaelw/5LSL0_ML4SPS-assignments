import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    """
    Calculate element-wise Rectified Linear Unit (ReLU)
    :param x: Input array
    :return: Rectified output
    """
    return np.maximum(x, 0)

def plot_Q10(X, y, h, save_fig=False):

    # plot x and h in separate plots
    # plt.figure(figsize=(12, 5))
    plt.rcParams['text.usetex'] = True
    max = 2.5
    min = -0.5

    # left plot: x
    # ax = plt.subplot(1, 2, 1)
    ax = plt.subplot(1,1,1)
    plt.scatter(X[(0, 3), 0], X[(0, 3), 1], s=160, facecolors='none', edgecolors='r')
    plt.scatter(X[(1, 2), 0], X[(1, 2), 1], s=160, marker='x', c='r')
    plt.xlim(min, max)
    plt.ylim(-0.5, 1.5)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    plt.grid(True)
    plt.xlabel(r'$\mathbf{x_1}$')
    plt.ylabel(r'$\mathbf{x_2}$')
    if save_fig:
        plt.savefig('Q-10-1.eps')
    plt.show()


    # right plot: h
    #ax = plt.subplot(1, 2, 2)
    ax = plt.subplot(1, 1, 1)
    plt.scatter(h[(0, 3), 0], h[(0, 3), 1], s=160, facecolors='none', edgecolors='r')
    plt.scatter(h[(1, 2), 0], h[(1, 2), 1], s=160, marker='x', c='r')
    plt.xlim(min, max)
    plt.ylim(-0.5, 1.5)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    plt.grid(True)
    plt.xlabel(r'$\mathbf{h_1}$')
    plt.ylabel(r'$\mathbf{h_2}$')
    # decision boundary f(x) = 0.5
    # Define x and y values
    x = [-0.5, 3.5]
    y = [-0.5, 1.5]
    plt.plot(x, y, linewidth=2, color='orange', label='$f(x) = 0.5$', linestyle='--')
    ax.legend()

    # save plot
    if save_fig:
        plt.savefig('Q-10-2.eps')

    plt.show()

def plot_Q11(X, y, h, save_fig=False):
    # plot x and h in separate plots
    plt.figure(figsize=(6, 4))
    plt.rcParams['text.usetex'] = True
    max = 1.5
    min = -0.4

    # left plot: x
    ax = plt.subplot(1, 1, 1)
    plt.scatter(X[(0, 3), 0], X[(0, 3), 1], s=160, facecolors='none', edgecolors='r')
    plt.scatter(X[(1, 2), 0], X[(1, 2), 1], s=160, marker='x', c='r')
    plt.xlim(min, max)
    plt.ylim(min, max)
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    plt.grid(True)
    plt.xlabel(r'$\mathbf{x_1}$')
    plt.ylabel(r'$\mathbf{x_2}$')

    # Define x and y values x2 = -x1 + 0.5
    x1 = [-4, 6]
    y1 = [4.5, -5.5]
    plt.plot(x1, y1, linewidth=2, color='cyan', linestyle='--', label='$x_2 = -x_1 + 0.5$')

    # Define x and y values x2 = -x1 + 1.5
    x2 = [4, -4]
    y2 = [-2.5, 5.5]
    plt.plot(x2, y2, linewidth=2, color='orange', linestyle='--', label='$x_2 = -x_1 + 1.5$')

    ax.legend()

    # save plot
    if save_fig:
        plt.savefig('Q-11.eps')

    plt.show()

def main():
    # define X, y for XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0, 1, 1, 0]]).T

    # define parameters
    W_1 = np.ones((2, 2))
    b_1 = np.array([[0, -1]]).T    
    h = np.zeros((4, 2))    

    # for every input x, calculate the output y
    for i, x in enumerate(X):
        h[[i]] = relu(np.dot(W_1, np.array([x]).T) + b_1).T

    # plot Q10
    plot_Q10(X, y, h, save_fig=True)

    # plot Q11
    plot_Q11(X, y, h, save_fig=True)
   

if __name__ == '__main__':
    main()