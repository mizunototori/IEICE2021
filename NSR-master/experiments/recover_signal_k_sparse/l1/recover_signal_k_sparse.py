import numpy as np
from nsr import NSR
from nsr import _make_nn_sparse_coded_signal
from nsr import _count_atoms
from matplotlib import pyplot as plt
import pickle

def l1_exam(n_nonzero_coefs):

    n_features = 20
    n_components = 50
    n_samples = 1500


    true_data, true_dictionary, true_code = \
        _make_nn_sparse_coded_signal(n_samples=n_samples,
                                     n_components=n_components,
                                     n_features=n_features,
                                     n_nonzero_coefs=n_nonzero_coefs,
                                     random_state=0)

    model = NSR(n_components=n_components, solver='mu',
                constraint='l1',
                alpha=0.2, max_iter=3000, tol=1e-5, verbose=1, eval_log=True)

    est_code = model.fit_transform(true_data, true_dict=true_dictionary)
    est_dictionary = model.dictionary
    est_logs = model.logs
    print("error: %r" % sum(sum(pow(true_data - np.dot(est_dictionary, est_code), 2))))

    atom_ratio, true_atoms, est_atoms = _count_atoms(est_dictionary, true_dictionary, axis=0, return_mat=True)

    print("recovery rate: %r" % atom_ratio)

    return atom_ratio
    '''
    figure_name = "wl1_over_complete_alpha_18e-1"

    plt.figure(1)
    plt.plot(est_logs['time'])
    plt.savefig('figures/' + figure_name + '_time.png')
    plt.figure(2)
    plt.plot(est_logs['error'])
    plt.savefig('figures/' + figure_name + '_error.png')
    plt.figure(3)
    plt.plot(est_logs['atoms'])
    plt.savefig('figures/' + figure_name + '_atoms.png')
    plt.figure(4)
    plt.plot(est_logs['sparsity'])
    plt.savefig('figures/' + figure_name + '_sparsity.png')


    plt.figure(5, figsize=(10, 10))
    print('len(true_atoms), len(est_atoms):', len(true_atoms), len(est_atoms))
    for i in range(1, len(true_atoms) + 1):
        plt.subplot(25, 2, i)
        print('idx:', i - 1)
        plt.plot(true_atoms[i - 1])
        plt.plot(est_atoms[i - 1])
        plt.savefig('figures/' + figure_name + '_compare_atoms.png')
    '''

if __name__ == '__main__':

    n_k = 13 
    n_iter = 1
    recv_rates = np.zeros((n_k - 3, n_iter))
    file_name = 'result_of_k_3_12_trial_1.pickle'

    for k in range(3, n_k):
        for i in range(0, n_iter):
            print('k, i:', k, i)
            print('k - 3, i', k-3, i)
            recv_rates[k - 3][i] = l1_exam(k)

    recv_rates
    with open(file_name, 'wb') as f:
        pickle.dump(recv_rates, f)
