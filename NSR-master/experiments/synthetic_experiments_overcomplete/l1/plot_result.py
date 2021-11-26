import pickle
import numpy as np
from matplotlib import pyplot as plt

constraint = 'l1'


def load_and_mean_est_logs(alpha, keyword):
    objs = []
    n_n_iter = 51
    for n_iter in range (1, n_n_iter):
        print('n_iter:', n_iter)
        file_name = "results/" + str(constraint) + "_alpha_" + str(alpha) + "_" + str(n_iter) + ".pickle"

        with open(file_name, 'rb') as f:
            obj = pickle.load(f)
            est_logs = obj['est_logs']
            if keyword is 'sparsity':
                mean_sp = np.mean(est_logs[keyword], axis=1)
                objs.append(mean_sp)
            else:
                objs.append(est_logs[keyword])
    return np.mean(objs, axis=0)


keywords = ['atoms', 'sparsity', 'time', 'error']
n_alpha = 10
log_means = []
# legend_tuple =  tuple([str(round(a * 0.1, 1)) for a in range(1, 10)])
legend_tuple =  tuple([str(round(a * 0.1, 1)) for a in range(7, n_alpha)])

for a in range(1, n_alpha):
    alpha = round(a * 0.1, 1)
    log_means.append(load_and_mean_est_logs(alpha, keywords[1]))

plt.figure(figsize=(8.5, 5))
lineObjects = plt.plot(np.array(log_means).T)
plt.legend(iter(lineObjects), legend_tuple, loc="center left", bbox_to_anchor=(1, 0.5), numpoints=1)
plt.show()






