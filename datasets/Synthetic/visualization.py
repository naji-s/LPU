import matplotlib.pyplot as plt
import numpy as np
import scipy.special
def plot_all_four(sig_X, l ,y, real_params):
    true_gamma, true_lambda, _, _, true_alpha, true_beta = real_params
    X_min = np.min(sig_X[:, 0]) * 3
    X_max = np.max(sig_X[:, 0]) * 3
    Y_min = np.min(sig_X[:, 1]) * 3
    Y_max = np.max(sig_X[:, 1]) * 3
    Y_grid = np.arange(Y_min, Y_max, 0.1)
    X_grid = np.arange(X_min, X_max, 0.1)
    X_grid, Y_grid = np.meshgrid(X_grid, Y_grid)
    p_l_y_x_Z = []
    a = 0
    b = 0
    for x_, y_ in zip(X_grid, Y_grid):
        psych_input = np.hstack([x_.reshape((-1, 1))-a, y_.reshape((-1, 1))-b])
        p_l_y_x_Z.append((scipy.special.expit(psych_input @ true_alpha + true_beta)) * (1 - true_gamma - true_lambda) + true_gamma)
    p_l_y_x_Z = np.asarray(p_l_y_x_Z).reshape(X_grid.shape)
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches((7.5, 7.5))
    ax[0, 0].scatter(sig_X[y==1, 0], sig_X[y==1, 1], alpha=0.8, c='b', marker='+')
    ax[0, 0].scatter(sig_X[y==0, 0], sig_X[y==0, 1], alpha=0.8, c='r', marker='_')
    ax[0, 0].contourf(X_grid, Y_grid, p_l_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.3)
    ax[0, 0].set_title(r'$p(X,l), l=$' + str(l.sum()) + r', $N=$'+str(l.shape[0]))

    ax[0, 1].scatter(sig_X[:, 0][l==1], sig_X[:, 1][l==1], alpha=0.8, c='b', marker='+')
    ax[0, 1].set_title(r'$p(X,l), l=$' + str(l.sum()) + r', $N=$'+str(l.shape[0]))
    ax[0, 1].contourf(X_grid, Y_grid, p_l_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.5)

    ax[1, 1].scatter(sig_X[:, 0][np.logical_and(l==1, y==1)], sig_X[:, 1][np.logical_and(l==1, y==1)], alpha=0.8, c='b', marker='+')
    ax[1, 1].scatter(sig_X[:, 0][np.logical_and(l==1, y==0)], sig_X[:, 1][np.logical_and(l==1, y==0)], alpha=0.8, c='b', marker='+')
    ax[1, 1].scatter(sig_X[:, 0][np.logical_and(l==0, y==1)], sig_X[:, 1][np.logical_and(l==0, y==1)], alpha=0.8, c='k', marker=r'$?$')
    ax[1, 1].scatter(sig_X[:, 0][np.logical_and(l==0, y==0)], sig_X[:, 1][np.logical_and(l==0, y==0)], alpha=0.8, c='k', marker=r'$?$')
    ax[1, 1].contourf(X_grid, Y_grid, p_l_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.5)
    ax[1, 1].set_title(r'$p(X,l), l=$' + str(l.sum()) + r', $N=$'+str(l.shape[0]))

    ax[1, 0].scatter(sig_X[:, 0][l==0], sig_X[:, 1][l==0], alpha=0.8, c='k', marker=r'$?$')
    ax[1, 0].contourf(X_grid, Y_grid, p_l_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.5)

    ax[1, 0].set_title(r'$p(X|l=0)$')
    ax[1, 0].set_xlabel(r'$X_1$')
    ax[1, 0].set_ylabel(r'$X_2$')
    ax[0, 1].set_xlabel(r'$X_1$')
    ax[0, 1].set_ylabel(r'$X_2$')
    ax[0, 0].set_xlabel(r'$X_1$')
    ax[0, 0].set_ylabel(r'$X_2$')
    ax[1, 1].set_xlabel(r'$X_1$')
    ax[1, 1].set_ylabel(r'$X_2$')


    cbar = fig.colorbar(ax[0, 0].contourf(X_grid, Y_grid, p_l_y_x_Z, cmap=plt.cm.RdYlBu, alpha=0.3))
    return fig