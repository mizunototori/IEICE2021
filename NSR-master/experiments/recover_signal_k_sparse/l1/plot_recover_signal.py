import pickle
from matplotlib import pyplot as plt


def plot_k_(data, title, fig_name, y_lim=None):
    plt.figure(figsize=(10, 6))
    plt.yticks(fontsize=15)
    plt.xticks(range(0, 14), range(1, 13), fontsize=15)
    if y_lim is not None:
        plt.ylim(*y_lim)
    xlabel_str = 'Number of non-zero elements of true sparse signal'
    plt.xlabel(xlabel_str, fontsize=18)

    plt.ylabel(title, fontsize=18)
    plt.plot(data, marker='x')
    plt.savefig('figures/' + fig_name)


n_k = 13
n_iter = 1
alpha = 0.2
p = 0.9
constraint = 'l1'
saved_data = []

for k in range(1, n_k):
    for i in range(0, n_iter):
        file_name = constraint + '_k_' + str(k) +\
            '_alpha_' + str(alpha) +\
            '_' + str(i) + '.pickle'
        with open(file_name, 'rb') as f:
            saved_data.append(pickle.load(f))

l2_error = [d['l2_error'] for d in saved_data]
n_iter = [d['n_iter'] for d in saved_data]
times = [d['times'] for d in saved_data]
# error = [d['error'] for d in saved_data]
# sparsity = [d['sparsity'] for d in saved_data]
atom_ratio = [d['atom_ratio'] for d in saved_data]
support_dist = [d['support_dist'] for d in saved_data]
code_ratio = [d['code_ratio'] for d in saved_data]

plot_k_(l2_error, '$\\ell_2$-error', 'l2_error', y_lim=(0, 1))
plot_k_(atom_ratio, 'Recovery ratio of Dictionary[%]', 'recovery_ratio')
plot_k_(times, 'Times', 'times')
plot_k_(n_iter, 'Number of Iterations', 'n_iter')
plot_k_(support_dist, 'Support Distance', 'support_dist', y_lim=(0, 1))
plot_k_(code_ratio, 'Recovery ratio of Coefficients [%]', 'code_ratio')



