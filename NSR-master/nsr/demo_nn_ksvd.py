import numpy as np
from nn_ksvd import NN_KSVD
from nsr import _make_nn_sparse_coded_signal
from sklearn.preprocessing import normalize

def nn_ksvd_exam():

    n_features = 20
    n_components = 50
    n_samples = 1500
    n_nonzero_coefs = 3
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
                 max_iter=2, tol=1e-7, verbose=1, eval_log=0)

    est_code = model.fit_transform(true_data)
    est_dictionary =  model.dictionary


if __name__ == '__main__':
    nn_ksvd_exam()