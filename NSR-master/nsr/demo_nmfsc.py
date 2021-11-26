import numpy as np
from sklearn.preprocessing import normalize
from nmfsc import NMFsc
from nsr_utils import _make_nn_sparse_coded_signal, _reorder_matrix, \
                      _support_dist, _l2_error, _count_codes, _count_atoms
import pickle

def exam():
    n_features = 20
    n_components = 50
    n_samples = 1500
    n_nonzero_coefs = 3
    sc_code = 0.85
    max_iter = 1000
    true_data, true_dictionary, true_code = \
        _make_nn_sparse_coded_signal(n_samples=n_samples,
                                     n_components=n_components,
                                     n_features=n_features,
                                     n_nonzero_coefs=n_nonzero_coefs,
                                     random_state=0)
    true_data /= np.max(true_data)
    model = NMFsc(n_components=n_components, solver='mu', sc_dict=None,
                  sc_code=sc_code, max_iter=max_iter, verbose=1, eval_log=True)

    est_code = model.fit_transform(true_data, true_dict=true_dictionary)
    est_dictionary = model.dictionary
    est_logs = model.logs

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

    file_name = 'nmfsc_sc_dict_None_sc_code_1'

    save_objects = {'true_dictionary': true_dictionary,
                    'true_code': true_code,
                    'est_dictionary': est_dictionary,
                    'est_code': est_code,
                    'est_logs': est_logs}

    with open(file_name + '.pickle', 'wb') as f:
        pickle.dump(save_objects, f)

if __name__ == '__main__':
    exam()

