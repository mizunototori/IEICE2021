import numpy as np
from nsr import NSR
from nsr import _make_nn_sparse_coded_signal
from sklearn.preprocessing import normalize
from nsr_utils import _count_atoms
from nsr_utils import _count_codes
from nsr_utils import _support
from nsr_utils import _reorder_matrix
from nsr_utils import _l2_error
from nsr_utils import _support_dist
from matplotlib import pyplot as plt
import pickle


def lp_exam(file_name, k=3, constraint='wl1', alpha=0.35, p=0.9):
    '''
    n_features = 5 # 20
    n_components = 10 # 50
    n_samples = 150 # 1500
    n_nonzero_coefs = 1
    '''
    n_features = 20
    n_components = 50
    n_samples = 1500
    n_nonzero_coefs = k
    true_data, true_dictionary, true_code = \
        _make_nn_sparse_coded_signal(n_samples=n_samples,
                                     n_components=n_components,
                                     n_features=n_features,
                                     n_nonzero_coefs=n_nonzero_coefs,
                                     random_state=0)

    model = NSR(n_components=n_components, solver='mu',
                constraint=constraint, p=p, e=0.0001,
                alpha=alpha, max_iter=40000, tol=1e-7, verbose=1, eval_log=True)

    est_code = model.fit_transform(true_data, true_dict=true_dictionary)
    est_dictionary = model.dictionary
    est_logs = model.logs
    # print("error: %r" % sum(sum(pow(true_data - np.dot(est_dictionary, est_code), 2))))

    atom_ratio, true_atoms, est_atoms, perm_idx = _count_atoms(est_dictionary, true_dictionary, axis=0, return_mat=True, return_idx=True)

    perm_est_code = _reorder_matrix(perm_idx, est_code)

    norm_true_code = normalize(true_code, axis=1)
    norm_est_code = normalize(perm_est_code, axis=1)

    sd = _support_dist(norm_true_code, norm_est_code)
    print("atom recovery rate: %r" % atom_ratio)

    l2error =  np.mean([_l2_error(t, e) for t, e in zip(norm_true_code, norm_est_code)])
    code_ratio = _count_codes(norm_true_code, norm_est_code)

    print("distance between supports: %r" % sd)
    print("l2 error: %r" % l2error)
    print("code recovery rate: %r" % code_ratio)




    save_objects = {'true_dictionary': true_dictionary,
                    'true_code': true_code,
                    'est_dictionary': est_dictionary,
                    'est_code': est_code,
                    'est_logs': est_logs}

    with open(file_name + '.pickle', 'wb') as f:
        pickle.dump(save_objects, f)

if __name__ == '__main__':
    constraint = 'wl1'
    alpha = 0.2
    p = 0.9
    k = 11

    for n in range(0, 1):
        file_name = 'result_lp/_' + constraint + '_k_' + str(k) +'_alpha_' + str(alpha) + '_p_' + str(p) + '_n_' + str(n)
        lp_exam(file_name, k=k, constraint=constraint, alpha=alpha, p=p)
