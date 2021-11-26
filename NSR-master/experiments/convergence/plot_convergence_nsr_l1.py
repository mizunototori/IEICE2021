import numpy as np
import pickle
import os
import glob
from matplotlib import pyplot as plt


def files2cost(target_dir, k):

    file = glob.glob(target_dir + '*k_' + str(k) + '_*.pickle')
    with open(file[0], 'rb') as f:
        result = pickle.load(f)['est_logs']['cost']
    print(len(result))
    i = np.argmin(result)
    print('Min cost[i]: ' + '[', str(i) + ']' + str(result[i]))
    return result, i


def plot_cost(file_name, cost, k):
    fig, ax = plt.subplots()
    ax.set_title('Cost of NN-MU-l1: k=' + str(k))
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Cost')
    ax.loglog(basey=10)
    ax.plot(cost)
    fig.savefig(file_name)

def calc_diff_tol(data, i):
    diff_tol = []
    for i in range(0, len(data)-1):
        diff_tol.append(abs(data[i] - data[i+1]) / data[i])
    print(diff_tol[i])
    return diff_tol


def plot_diff_tol(file_name, diff_tol, k):
    fig, ax = plt.subplots()
    ax.set_title('Difference of tolerance of NN-MU-l1: k=' + str(k))
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Difference of tolerance')
    ax.loglog(basey=10)
    ax.plot(diff_tol)
    fig.savefig(file_name)


def main():
    k =11
    file_name = 'fig_convergence_nsr_l1_k_' + str(k)
    cost, i = files2cost('result_l1/', k)
    plot_cost(file_name + '.png', cost, k)
    diff_tol = calc_diff_tol(cost, i)
    plot_diff_tol(file_name + '_tol.png', diff_tol, k)

if __name__ == '__main__':
    main()
