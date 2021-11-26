import numpy as np
from nsr import NSR
import pickle
from nsr import _make_nn_sparse_coded_signal
from nsr import _count_atoms
from matplotlib import pyplot as plt

def l1_exam(alpha, constraint, file_name):
    n_features = 20
    n_components = 50
    n_samples = 1500
    n_nonzero_coefs = 3


    true_data, true_dictionary, true_code = \
        _make_nn_sparse_coded_signal(n_samples=n_samples,
                                     n_components=n_components,
                                     n_features=n_features,
                                     n_nonzero_coefs=n_nonzero_coefs,
                                     random_state=0)

    # true_data, true_dictionary, true_code = create_data(n_features, n_components, n_samples, n_nonzero_coefs)

    model = NSR(n_components=n_components, solver='mu',
                constraint=constraint, alpha=alpha,
                max_iter=2000, tol=1e-14, verbose=1, eval_log=True)

    est_code = model.fit_transform(true_data, true_dict=true_dictionary)
    est_dictionary = model.dictionary
    est_logs = model.logs

    atom_ratio, true_atoms, est_atoms = _count_atoms(est_dictionary, true_dictionary, axis=0, return_mat=True)

    print("recovery rate: %r" % atom_ratio)

    save_objects = {'true_data': true_data,
                    'true_dictionary': true_dictionary,
                    'true_code': true_code,
                    'est_dictionary': est_dictionary,
                    'est_code': est_code,
                    'est_logs': est_logs}

    with open(file_name, 'wb') as f:
        pickle.dump(save_objects, f)


if __name__ == '__main__':
    constraint = 'l1'
    for a in range(1, 10):
        alpha = round(a * 0.1, 1)
        n_iter = 50
        file_name = "results/" + str(constraint) + "_alpha_" + str(alpha) + "_" + str(n_iter) + ".pickle"
        l1_exam(alpha, constraint, file_name)
