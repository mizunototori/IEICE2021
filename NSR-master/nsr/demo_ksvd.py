import numpy as numpy
from ksvd import KSVD
from sklearn.datasets import make_sparse_coded_signal
from sklearn.preprocessing import normalize
import random

def ksvd_exam():

    n_features = 20
    n_components = 50
    n_samples = 1500
    n_nonzero_coefs = 3
    sigma = 0.1

    true_data, true_dictionary, true_code = \
        make_sparse_coded_signal(n_samples=n_samples,
                                     n_components=n_components,
                                     n_features=n_features,
                                     n_nonzero_coefs=n_nonzero_coefs,
                                     random_state=random.randint(0, 99))

    model = KSVD(n_components=n_components, sigma=sigma,
                 n_nonzero_coefs=n_nonzero_coefs, solver='omp',
                 true_dict=true_dictionary,
                 max_iter=30, tol=1e-7, verbose=1, eval_log=True)

    est_code = model.fit_transform(true_data)
    est_dictionary =  model.dictionary


if __name__ == '__main__':
    ksvd_exam()