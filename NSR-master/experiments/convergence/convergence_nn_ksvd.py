import numpy as np
from nn_ksvd import NN_KSVD
from nsr import _make_nn_sparse_coded_signal
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import pickle
def nn_ksvd_exam(file_name, k=3):

    n_features = 20
    n_components = 50
    n_samples = 1500
    n_nonzero_coefs = k
    sigma = 0.1

    true_data, true_dictionary, true_code = \
        _make_nn_sparse_coded_signal(n_samples=n_samples,
                                     n_components=n_components,
                                     n_features=n_features,
                                     n_nonzero_coefs=n_nonzero_coefs,
                                     random_state=0)

    model = NN_KSVD(n_components=n_components, sigma=sigma,
                 n_nonzero_coefs=n_nonzero_coefs, solver='omp',
                 true_dict=true_dictionary,
                 max_iter=40000, tol=1e-7, verbose=1, eval_log=True)

    est_code = model.fit_transform(true_data)
    est_dictionary =  model.dictionary
    est_logs = model.logs

    save_objects = {'est_logs': est_logs}

    with open(file_name + '.pickle', 'wb') as f:
        pickle.dump(save_objects, f)

if __name__ == '__main__':
    k = 11
    for n in range(0, 1):
        file_name = 'result_nnksvd/' +'nn_ksvd_' + '_k_' + str(k) + '_n_' + str(n)
        nn_ksvd_exam(file_name, k=k)
