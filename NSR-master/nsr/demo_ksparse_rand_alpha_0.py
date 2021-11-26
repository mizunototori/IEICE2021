import numpy as np
from nn_ksvd import NN_KSVD
from nsr import NSR
from nsr import _make_nn_sparse_coded_signal
from sklearn.preprocessing import normalize
import pickle
import time

from nsr import _count_atoms
from nsr import _support
from nsr import _reorder_matrix
from nsr import _l2_error
from nsr import _support_dist

def nn_exam(k, model, model_name):

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
                                     random_state=0,
                                     randn_flag=False,
                                     alpha=0)

    start_time = time.time()
    est_code, n_iter = model.fit_transform(true_data, return_iter=True)
    end_time = time.time()

    # Time
    during_time = end_time - start_time

    # Approximation Error
    est_dictionary = model.dictionary
    error = np.linalg.norm(true_data - np.dot(est_dictionary, est_code), ord='fro') / np.prod(true_code.shape)
    
    # Recovery ratio
    atom_ratio, true_atoms, est_atoms, perm_idx = _count_atoms(est_dictionary, true_dictionary, axis=0, return_mat=True, return_idx=True)

    # Support distance
    perm_est_code = _reorder_matrix(perm_idx, est_code)
    norm_true_code = normalize(true_code, axis=1)
    norm_est_code = normalize(perm_est_code, axis=1)
    sd = _support_dist(norm_true_code, norm_est_code)

    # L2 error
    l2error =  np.mean([_l2_error(t, e) for t, e in zip(norm_true_code, norm_est_code)])
    
    result_obj = {'n_nonzero_coefs':n_nonzero_coefs, 'model_name': model_name, 'atoms':atom_ratio, 'support_dis': sd, 'approx_err': error, 'l2_err': l2error, 'time':during_time, 'iter':n_iter}

    return result_obj

if __name__ == '__main__':



    file_name = 'pilot_study_ksparse_rand_alpha_0.pickle'
    results = []

    for k in range(1, 12):
        print(k)
        n_features = 20
        n_components = 50
        n_samples = 1500
        sigma = 0.1
        ksvd_maxiter = 3000
        nsr_maxiter = 10000
        ksvd_tol = 1e-5
        nsr_tol = 1e-7

        model_names = [ r"NN-MU-$\ell_p (p=0.9)$", r"NN-MU-$\ell_1$", 'NN-KSVD']

        models = []

        models.append(NSR(n_components=n_components, solver='mu',
                    constraint='wl1', p=0.9, e=0.01, alpha=0.25,
                    max_iter=nsr_maxiter, tol=nsr_tol, verbose=0, eval_log=False))
        models.append(NSR(n_components=n_components, solver='mu',
                constraint='l1',
                alpha=0.2, max_iter=nsr_maxiter, tol=nsr_tol, verbose=0, eval_log=False))

        models.append(NN_KSVD(n_components=n_components, sigma=sigma,
                 n_nonzero_coefs=k, solver='omp',
                 max_iter=ksvd_maxiter, tol=ksvd_tol, verbose=0, eval_log=False))

        for i, model in enumerate(models):
            print(model_names[i])
            results.append(nn_exam(k, model, model_names[i]))


    with open(file_name, 'wb') as f:
        pickle.dump(results, f)









