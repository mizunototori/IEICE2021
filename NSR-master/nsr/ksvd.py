# ref: https://github.com/kibo35/sparse-modeling/blob/master/ch12.ipynb
# import libraries
import numpy as np
import time
from sklearn.preprocessing import normalize
from nsr import _calc_error
from nsr import _count_atoms
from nsr import _hoyers_sparsity


def _evaluations(data, dictionary, code, true_dict=None):
    _time = time.time()
    _error = _calc_error(data, dictionary, code)
    _sparsity = _hoyers_sparsity(normalize(code, axis=1))

    if true_dict is not None:
        _atoms = _count_atoms(dictionary, true_dict)
        return _time, _error, _sparsity, _atoms
    else:
        return _time, _error, _sparsity


def _omp(y, D, n_nonzero_coefs, tol):
    # y = D x
    n_features, n_components = D.shape

    S = np.zeros(n_components)
    x = np.zeros(n_components)

    r = y.copy()
    rr = np.dot(r, r)

    for _ in range(n_nonzero_coefs):
        error = rr - np.dot(D[:, S == 0].T, r) ** 2

        idx = np.where(S == 0)[0]
        S[idx[error.argmin()]] = 1

        Ds = D[:, S == 1]
        pinv = np.linalg.pinv(np.dot(Ds, Ds.T))
        x[S == 1] = np.dot(Ds.T, np.dot(pinv, y))

        r = y - np.dot(D, x)
        rr = np.dot(r, r)
        if rr < tol:
            break
    return x


def _clear_dictionary(dictionary, code, data, n_nonzero_coefs=3):
    n_features, n_components = dictionary.shape
    n_components, n_samples = code.shape
    norms = np.sqrt(sum(dictionary ** 2))
    norms = norms[:, np.newaxis].T
    dictionary = dictionary / np.dot(np.ones((n_features, 1)), norms)
    code = code * np.dot(norms.T, np.ones((1, n_samples)))

    t1 = n_nonzero_coefs
    t2 = 0.99

    error = sum((data - np.dot(dictionary, code)) ** 2)
    gram = np.dot(dictionary.T, dictionary)
    gram = gram - np.diag(np.diag(gram))

    for i in range(0, n_components):
        if (max(gram[i, :]) > t2) or \
                (len(*np.nonzero(abs(code[i, :]) > 1e-7)) <= t1):
            val = np.max(error)
            pos = np.argmax(error)
            error[pos] = 0
            dictionary[:, i] = data[:, pos] / np.linalg.norm(data[:, pos])
            gram = np.dot(dictionary.T, dictionary)
            gram = gram - np.diag(np.diag(gram))

    return dictionary


def _svd_rank_approximation(data, dictionary, code, update_idx):

    code_using = code[update_idx, :] != 0
    residual_err = \
        data[:, code_using] - np.dot(dictionary, code[:, code_using])

    if np.sum(code_using) == 0:
        error = data - np.dot(dictionary, code)
        norm_error = sum(error)
        max_idx = np.argmax(norm_error)
        new_dictionary = data[:, max_idx]
        new_dictionary = \
            new_dictionary / np.sqrt(np.dot(new_dictionary.T, new_dictionary))
        new_dictionary = new_dictionary * np.sign(new_dictionary)

        return new_dictionary, code[update_idx, :]

    code[update_idx, code_using] = 0
    residual_err = \
        data[:, code_using] - np.dot(dictionary, code[:, code_using])

    U, s, Vt = np.linalg.svd(residual_err)
    dictionary[:, update_idx] = U[:, 0]
    code[update_idx, code_using] = s[0] * Vt.T[:, 0]

    update_dictionary = dictionary[:, update_idx]
    update_code = code[update_idx, code_using]

    return update_dictionary, update_code


def _ksvd_omp_update(data, sigma, n_components, n_nonzero_coefs, max_iter=50,
                     true_dict=None, initial_dict=None, tol=1e-4, verbose=0,
                     eval_log=None):

    n_features, n_samples = data.shape

    # Initialize dictionary and code
    if initial_dict is None:
        dictionary = data[:, :n_components]
        dictionary = \
            np.dot(dictionary, np.diag(1. / np.sqrt(np.diag(np.dot(dictionary.T, dictionary)))))
    else:
        dictionary = initial_dict.copy()

    code = np.zeros((n_components, n_samples))
    tol = n_features * (sigma ** 2)

    for n_iter in range(1, max_iter + 1):

        if true_dict is not None:
            time_lps, error, sparsity, atoms = _evaluations(data, dictionary, code, true_dict=true_dict)
        else: 
            time_lps, error, sparsity = _evaluations(data, dictionary, code, true_dict=None)

        start_time = time_lps
        error_at_init = error
        previous_error = error_at_init

        time_log = np.zeros(max_iter)
        error_log = np.zeros(max_iter)
        sparse_log = np.zeros(max_iter)
        cost_log = np.zeros(max_iter)

        if true_dict is not None:
            atom_at_init = atoms
            previous_atom = atom_at_init

            atom_log = np.zeros(max_iter)


        for i in range(n_samples):
            c = _omp(data[:, i], dictionary, n_nonzero_coefs, tol=tol)
            code[:, i] = c
        '''
        for j in range(n_components):
            code_using = code[j, :] != 0
            if sum(code_using) <  2: # if the size of code_using is 0 or 1, it can't be SVD.
                continue
            else:
                d, c = _svd_rank_approximation(data, dictionary, code, j)
                dictionary[:, j] = d
                code[j, code_using] = c
        '''
        for j in range(n_components):
            code_using = code[j, :] != 0
            if np.sum(code_using) == 0:
                continue
            code[j, code_using] = 0
            residual_err = data[:, code_using] - np.dot(dictionary, code[:, code_using])
            U, s, Vt = np.linalg.svd(residual_err)
            dictionary[:, j] = U[:, 0]
            code[j, code_using] = s[0] * Vt.T[:, 0]

        #dictionary = _clear_dictionary(dictionary, code, data, n_nonzero_coefs=n_nonzero_coefs)

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

        if verbose:

            ref_tol = (previous_error - error) / previous_error
            if verbose:
                iter_time = time.time()
                if true_dict is not None and eval_log:
                    print("Epoch %02d reached after %.3f seconds, error: %f, sparsity (mean): %f, atom: %f" %
                          (n_iter, iter_time - start_time, error, np.mean(logged_sp), logged_atom))
                else:
                    print("Epoch %02d reached after %.3f seconds, error: %f, sparsity (mean): %f" %
                          (n_iter, iter_time - start_time, error, np.mean(sparsity)))
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

def _ksvd(data, sigma, n_components, n_nonzero_coefs,
                      solver='omp', true_dict=None, initial_dict=None,
                      max_iter=50, tol=1e-4,
                      verbose=0, eval_log=None):
    if solver == 'omp':
        if eval_log is None:
            dictionary, code, n_iter = \
                _ksvd_omp_update(data, sigma, n_components, n_nonzero_coefs,
                                    tol=tol, max_iter=max_iter,
                                    true_dict=true_dict, initial_dict=initial_dict,
                                    verbose=verbose,
                                    eval_log=eval_log)
            return dictionary, code, n_iter

        else:
            dictionary, code, n_iter, logs = \
                _ksvd_omp_update(data, sigma, n_components, n_nonzero_coefs,
                                    tol=tol, max_iter=max_iter,
                                    true_dict=true_dict, initial_dict=initial_dict,
                                    verbose=verbose,
                                    eval_log=eval_log)
            return dictionary, code, n_iter, logs

    else:
            raise ValueError("Invalid solver parameter '%s'." % solver)



class KSVD(object):
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

    def fit_transform(self, data, true_dict=None, return_iter=None, initial_dict=None):
        if self.eval_log is None:

            dictionary, code, n_iter = \
                _ksvd(data=data,
                                  sigma=self.sigma,
                                  n_components=self.n_components,
                                  n_nonzero_coefs=self.n_nonzero_coefs,
                                  tol=self.tol,
                                  solver=self.solver,
                                  true_dict=self.true_dict,
                                  initial_dict=initial_dict,
                                  max_iter=self.max_iter,
                                  verbose=self.verbose,
                                  eval_log=self.eval_log)
        else:
            dictionary, code, n_iter, logs= \
                _ksvd(data=data,
                                  sigma=self.sigma,
                                  n_components=self.n_components,
                                  n_nonzero_coefs=self.n_nonzero_coefs,
                                  tol=self.tol,
                                  solver=self.solver,
                                  true_dict=self.true_dict,
                                  initial_dict=initial_dict,
                                  max_iter=self.max_iter,
                                  verbose=self.verbose,
                                  eval_log=self.eval_log)
            self.logs = logs

        self.dictionary = dictionary

        if return_iter:
            return code, n_iter
        else:
            return code