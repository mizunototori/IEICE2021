# ref: https://github.com/kibo35/sparse-modeling/blob/master/ch12.ipynb
# import libraries
import numpy as np
import sys
#from omp import omp as nn_omp
from pyomp.omp import omp as nn_omp
from scipy.optimize import nnls
import time
from sklearn.preprocessing import normalize
from sklearn.utils import check_array
from scipy.sparse.linalg import svds
from nsr_utils import _calc_error
from nsr_utils import _count_atoms
from nsr_utils import _hoyers_sparsity
from ksvd import _clear_dictionary

def _evaluations(data, dictionary, code, true_dict=None):
    _time = time.time()
    _error = _calc_error(data, dictionary, code)
    _sparsity = _hoyers_sparsity(normalize(code, axis=0))

    if true_dict is not None:
        _atoms = _count_atoms(dictionary, true_dict)
        return _time, _error, _sparsity, _atoms
    else:
        return _time, _error, _sparsity
def _calc_approx1(residual_err, _u, _s, _v):
    u = _u.copy()
    s = _s.copy()
    v = _v.copy()
    u[u < 0] = 0
    v[v < 0] = 0
    approx = np.linalg.norm(residual_err - np.dot(u, s*v.T))
    return approx

def _calc_approx2(residual_err, _u, _s, _v):
    u = _u.copy()
    s = _s.copy()
    v = _v.copy()
    u *= np.sign(u)
    v *= np.sign(v)
    approx = np.linalg.norm(residual_err - np.dot(u, s*v.T))
    return approx

def _svd_rank_approximation(data, dictionary, code, update_idx):

    code_using = code[update_idx, :] != 0
    

    if np.sum(code_using) == 0:
        error = data - np.dot(dictionary, code)
        norm_error = sum(error)
        max_idx = np.argmax(norm_error)
        new_dictionary = data[:, max_idx]
        new_dictionary = new_dictionary / np.sqrt(np.dot(new_dictionary.T, new_dictionary))
        #new_dictionary = new_dictionary * np.sign(new_dictionary)
        new_dictionary[new_dictionary < 0]  = 1e-14
        return new_dictionary, code[update_idx, :]


    reduced_code = code[:, code_using]
    reduced_data = data[:, code_using]
    reduced_code[update_idx, :] = 0

    residual_err = reduced_data - np.dot(dictionary, reduced_code)

    u1, s1, v1t = svds(residual_err, 1)

    v1 = v1t.T

    approx1 = _calc_approx1(residual_err, u1, s1, v1)
    approx2 = _calc_approx2(residual_err, u1, s1, v1)

    if (approx1 < approx2):
        u1[u1 < 0] = 1e-14
        v1[v1 < 0] = 1e-14
    else:

        u1 *= np.sign(u1)
        v1 *= np.sign(v1)

    update_dictionary = u1
    update_code = s1 * v1

    dict_norm = np.sqrt(np.dot(update_dictionary.T, update_dictionary))
    update_dictionary = update_dictionary / dict_norm
    update_code = np.dot(update_code, dict_norm)

    update_dictionary = check_array(update_dictionary.reshape(1, -1))
    update_code = check_array(update_code.reshape(1, -1))

    return update_dictionary, update_code

def _nn_ksvd_omp_update(data, sigma, n_components, n_nonzero_coefs, max_iter=50,
                        true_dict=None, tol=1e-4, verbose=0, eval_log=None):

    # Check arrays
    # data = check_array(data)

    n_features, n_samples = data.shape

    # Initialize data, dictionary and code
    # data = normalize(data, axis=0)
    dictionary = data[:, :n_components]

    dictionary = np.dot(dictionary, np.diag(1. / np.sqrt(np.diag(np.dot(dictionary.T, dictionary)))))

    code = np.zeros((n_components, n_samples))
    #tol = n_features * (sigma ** 2)

    if true_dict is not None:
        time_lps, error, sparsity, atoms = _evaluations(data, dictionary, code, true_dict=true_dict)
    else: 
        time_lps, error, sparsity = _evaluations(data, dictionary, code, true_dict=None)

    start_time = time_lps
    error_at_init = error
    previous_error = error_at_init
    previous2_error = error_at_init


    time_log = np.zeros(max_iter)
    error_log = np.zeros(max_iter)
    sparse_log = np.zeros(max_iter)
    cost_log = np.zeros(max_iter)

    if true_dict is not None:
        atom_at_init = atoms
        previous_atom = atom_at_init

        atom_log = np.zeros(max_iter)

    for n_iter in range(1, max_iter + 1):


        for i in range(n_samples):
                _c = nn_omp(dictionary, data[:, i], nonneg=True, ncoef=n_nonzero_coefs, tol=tol, verbose=False)
                c = _c.coef
                code[:, i] = c

        for j in range(n_components):
            code_using = code[j, :] != 0
            if sum(code_using) <  2: # if the size of code_using is 0 or 1, it can't be SVD.
                continue
            else:
                d, c = _svd_rank_approximation(data, dictionary, code, j)
                dictionary[:, j] = d
                code[j, code_using] = c

        dictionary = _clear_dictionary(dictionary, code, data)

        if eval_log:
            if true_dict is not None:
                time_lps, error, sparsity, atoms = _evaluations(data, dictionary, code, true_dict=true_dict)
            else: 
                time_lps, error, sparsity = _evaluations(data, dictionary, code, true_dict=None)

            logged_error = error
            logged_sp = sparsity
            iter_time = time_lps
            logged_time = iter_time - start_time

            error_log[n_iter - 1] = logged_error
            sparse_log[n_iter - 1] = np.mean(logged_sp)
            time_log[n_iter - 1] = logged_time

            if true_dict is not None:
                logged_atom = atoms
                atom_log[n_iter - 1] = logged_atom

        error = _calc_error(data, dictionary, code)
        ref_tol = (previous_error - error) / previous_error

        #if error == previous2_error: print(error, '!\n')

        if abs(ref_tol) < tol :#or error == previous2_error:
            break

        if verbose:
            iter_time = time.time()
            if true_dict is not None and eval_log:
                print("Epoch %02d reached after %.3f seconds, error: %f, sparsity (mean): %f, atom: %f" %
                      (n_iter, iter_time - start_time, error, np.mean(logged_sp), logged_atom))
            else:
                print("Epoch %02d reached after %.3f seconds, error: %f, sparsity (mean): %f" %
                      (n_iter, iter_time - start_time, error, np.mean(sparsity)))

        previous2_error = previous_error
        previous_error = error

    if eval_log is not None:
        if true_dict is not None:
            logs = {'time': time_log, 'error': error_log, 'cost': cost_log,
                    'sparsity': sparse_log, 'atoms': atom_log}
        else:
            logs = {'time': time_log, 'error': error_log, 'cost': cost_log,
                    'sparsity': sparse_log}
        return dictionary, code, n_iter, logs
    else:
        return dictionary, code, n_iter

def _nn_ksvd(data, sigma, n_components, n_nonzero_coefs,
                      solver='omp', true_dict=None,
                      max_iter=50, tol=1e-4,
                      verbose=0, eval_log=None):
    if solver == 'omp':
        if eval_log is None:
            dictionary, code, n_iter = \
                _nn_ksvd_omp_update(data, sigma, n_components, n_nonzero_coefs,
                                    tol=tol, max_iter=max_iter,
                                    true_dict=true_dict, verbose=verbose,
                                    eval_log=eval_log)
            return dictionary, code, n_iter

        else:
            dictionary, code, n_iter, logs = \
                _nn_ksvd_omp_update(data, sigma, n_components, n_nonzero_coefs,
                                    tol=tol, max_iter=max_iter,
                                    true_dict=true_dict, verbose=verbose,
                                    eval_log=eval_log)
            return dictionary, code, n_iter, logs

    else:
            raise ValueError("Invalid solver parameter '%s'." % solver)



class NN_KSVD(object):
    def __init__(self, sigma, n_components, n_nonzero_coefs, tol=1e-4,
                 solver='omp', true_dict=None,
                 max_iter=1000, verbose=0, eval_log=None):
        self.sigma=sigma
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.solver = solver
        self.true_dict = true_dict
        self.max_iter = max_iter
        self.verbose = verbose
        self.eval_log = eval_log

    def fit_transform(self, data, true_dict=None, return_iter=None):
        if self.eval_log is None:

            dictionary, code, n_iter = \
                _nn_ksvd(data=data,
                                  sigma=self.sigma,
                                  n_components=self.n_components,
                                  n_nonzero_coefs=self.n_nonzero_coefs,
                                  tol=self.tol,
                                  solver=self.solver,
                                  true_dict=self.true_dict,
                                  max_iter=self.max_iter,
                                  verbose=self.verbose,
                                  eval_log=self.eval_log)
        else:
            dictionary, code, n_iter, logs= \
                _nn_ksvd(data=data,
                                  sigma=self.sigma,
                                  n_components=self.n_components,
                                  n_nonzero_coefs=self.n_nonzero_coefs,
                                  tol=self.tol,
                                  solver=self.solver,
                                  true_dict=self.true_dict,
                                  max_iter=self.max_iter,
                                  verbose=self.verbose,
                                  eval_log=self.eval_log)
            self.logs = logs

        self.dictionary = dictionary

        if return_iter:
            return code, n_iter
        else:
            return code