import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import spdiags
import warnings
import time
from numpy.linalg import norm as linnorm
from nsr_utils import _make_nn_sparse_coded_signal, _hoyers_sparsity, \
                      _reorder_matrix, _support, _support_dist, _l2_error, \
                      _calc_error, _count_codes, _count_atoms #, _projfunc

from projfunc import projfunc as _projfunc

"""Decompose data signal Y as a dictionary D and sparse code X.
    Returns a dictionary D, and sparse code X
    Y (n_features, n_samples)
    D (n_features x n_components)
    X (n_components, n_samples)

"""
EPSILON = np.finfo(np.float32).eps
def _error(Y, A, X):
    return 0.5 * sum(sum((Y - np.dot(A, X)) ** 2))

def _multcols(A, b):
    return A * spdiags(b.T, 0, len(b), len(b))


def _normcols(x):
   return  _multcols(x, 1./np.sqrt(np.sum(x **2, axis=0)))


def _initialize_nmf(Y, n_components):
    n_features, n_samples = Y.shape

    if (n_features < n_components):
        warnings.warn("NMF expects complete/under-complete dictionary;"
                      "The dimension should be "
                      "n_components %r < n_features %r"
                      % (n_components, n_features))

    D = Y[:, :n_components].copy()
    X = np.random.ranf((n_components, n_samples))
    X = X/ (np.sqrt(np.sum(X ** 2)) * np.ones((1, n_samples)))
    return D, X


def _evaluations(data, dictionary, code, true_dict=None):
    _time = time.time()
    _error = _calc_error(data, dictionary, code)
    _sparsity = _hoyers_sparsity(normalize(code, axis=1))

    if true_dict is not None:
        _atoms = _count_atoms(_normcols(dictionary), _normcols(true_dict))
        return _time, _error, _sparsity, _atoms
    else:
        return _time, _error, _sparsity

"""
def _multiplicative_update_dict(data, dictionary, code, begobj, L1_dict=None,
                                stepsize_dict=1, verbose=0):
    n_features, n_components = dictionary.shape
    if L1_dict is not None:
        delta_dict = np.dot(np.dot(dictionary, code) - data, code.T)
        begobj = 0.5 * sum(sum(data - np.dot(dictionary, code) ** 2))
        while(True):
            new_dict = dictionary - stepsize_dict * delta_dict
            norms = np.sqrt(sum(new_dict ** 2))

            for i in range(0, n_components):
                tmp_dict, _ = _projfunc(new_dict[:, i][:, np.newaxis], L1_dict * norms[i],
                                        norms[i] ** 2, nn=True,
                                        verbose=verbose)
                new_dict[:, i] = tmp_dict.reshape(-1)

            newobj = 0.5 * sum(sum(data - np.dot(new_dict, code) ** 2))

            if newobj <= begobj:
                break

            stepsize_code /= 2

            if stepsize_code < 1e-200:
                if verbose:
                    print('Algorithm converged')
                return dictionary, stepsize_dict

            stepsize_dict = stepsize_dict * 1.2
            dictionary = new_dict
        return dictionary, stepsize_dict

    else:
        dictionary = dictionary * (np.dot(data, code.T)) / \
                    (np.dot(dictionary, np.dot(code, code.T)) + EPSILON)
        return dictionary, None


def _multiplicative_update_code(data, dictionary, code, begobj, L1_code=None,
                                stepsize_code=1, verbose=0):

    n_features, n_components = dictionary.shape
    if L1_code is not None:
        delta_code = np.dot(dictionary.T, (np.dot(dictionary, code) - data))

        while(True):
            new_code = code - stepsize_code * delta_code
            for i in range(0, n_components):
                tmp_code, _ = _projfunc(new_code[i, :][:, np.newaxis], L1_code, 1,
                                          nn=True, verbose=verbose)
                new_code[i, :] = tmp_code.T

            newobj = _calc_error(data, dictionary, new_code)

            if newobj <= begobj:
                break

            stepsize_code /= 2

            if stepsize_code < 1e-200:
                if verbose:
                    print('Algorithm converged')
                return code, stepsize_code

            stepsize_code = stepsize_code * 1.2
            code = new_code
        return code, stepsize_code

    else:
        code = code * (np.dot(dictionary.T, data)) / \
            (np.dot(np.dot(dictionary.T, dictionary), code) + EPSILON)

        return code, None


def _multiplicative_update(data, n_components, sc_dict=None, sc_code=None,
                           max_iter=1000, tol=1e-4, true_dict=None, verbose=0,
                           eval_log=None):
    # initial step size
    stepsize_dict = 1
    stepsize_code = 1

    n_features, n_samples = data.shape
    # initialize dictionary and code
    _data = data.copy()
    data = normalize(data, axis=0)
    dictionary, code = _initialize_nmf(data, n_components)

    if sc_dict is not None:
        L1_dict = n_components ** 0.5 - (n_components ** 0.5 - 1) * sc_dict
        for i in range(0, n_components):
            tmp_dict, _ = _projfunc(dictionary[:, i][:, np.newaxis], L1_dict, 1, nn=True)
            dictionary[:, i] = tmp_dict.reshape(-1)
    else:
        L1_dict = None

    if sc_code is not None:
        L1_code = n_samples ** 0.5 - (n_samples ** 0.5 - 1) * sc_code
        for i in range(0, n_components):
            tmp_code, _ = _projfunc(code[i, :][:, np.newaxis], L1_code, 1, nn=True)
            code[i, :] = tmp_code.reshape(-1)
    else:
        L1_code = None

    # initialize evaluation log

    if eval_log is True and true_dict is not None:
        time_lps, error, sparsity, atoms = \
            _evaluations(_data, dictionary, code, true_dict=true_dict)
    else:
        time_lps, error, sparsity = \
            _evaluations(_data, dictionary, code, true_dict=None)

    start_time = time_lps
    time_log = np.zeros(max_iter)
    error_log = np.zeros(max_iter)
    sparse_log = np.zeros(max_iter)

    if eval_log is True and true_dict is not None:
        atom_at_init = atoms
        previous_atom = atom_at_init

        atom_log = np.zeros(max_iter)

    for n_iter in range(1, max_iter + 1):

        begobj = _calc_error(data, dictionary, code)
        # Update code

        code, stepsize_code = _multiplicative_update_code(data, dictionary,
                                                          code, begobj,
                                                          L1_code=L1_code,
                                                          stepsize_code=stepsize_code)

        dictionary, stepsize_dict = _multiplicative_update_dict(data,
                                                                dictionary,
                                                                code, begobj,
                                                                L1_dict=L1_dict,
                                                                stepsize_dict=stepsize_dict)

        n_features, n_samples = data.shape
        norms = np.sqrt(sum(code.T ** 2))
        norms = norms[:, np.newaxis].T
        code = code / np.dot(norms.T, np.ones((1, n_samples)))
        dictionary = dictionary * np.dot(np.ones((n_features, 1)), norms)

        if eval_log:
            if true_dict is not None:
                time_lps, error, sparsity, atoms = _evaluations(_data, dictionary, code, true_dict=true_dict)
            else: 
                time_lps, error, sparsity = _evaluations(_data, dictionary, code, true_dict=None)


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

        if n_iter % 10 == 0:

            error = np.linalg.norm(data - np.dot(dictionary, code), ord='fro')
            sp = _hoyers_sparsity(normalize(code, axis=1))

            if verbose:
                iter_time = time.time()

                if true_dict is not None and eval_log:
                    print("Epoch %02d reached after %.3f seconds, error: %f, sparsity (mean): %f, atom: %f" %
                          (n_iter, iter_time - start_time, error, np.mean(logged_sp), logged_atom))
                else:
                    print("Epoch %02d reached after %.3f seconds, error: %f, sparsity (mean): %f" %
                          (n_iter, iter_time - start_time, error, np.mean(sp)))

    if eval_log is None:
        return dictionary, code, n_iter

    else:
        if eval_log is True and true_dict is not None:
            logs = {'time': time_log, 'error': error_log,
                    'sparsity': sparse_log, 'atoms': atom_log}
        else:
            logs = {'time': time_log, 'error': error_log,
                    'sparsity': sparse_log}

        return dictionary, code, n_iter, logs

"""
def _multiplicative_update(data, n_components, sc_dict=None, sc_code=None, max_iter=1000, tol=1e-4, true_dict=None, verbose=0, eval_log=None):

    """
    initialization
    """
    data = data / np.max(data)
    n_features, n_samples = data.shape
    dictionary = np.random.rand(n_features, n_components)
    code = np.random.rand(n_components, n_samples)

    data = data / np.max(data)
    code = code / (np.sqrt(sum(code ** 2)) * np.ones((1, n_samples)))


    if sc_dict is not None:
        L1_dict = n_features ** 0.5 - (n_features ** 0.5 - 1) * sc_dict
        for i in range(0, n_components):
            tmp_dict, _ = _projfunc(dictionary[:, i][:, np.newaxis], L1_dict, 1, nn=True)
            dictionary[:, i] = tmp_dict.reshape(-1)
    else:
        L1_dict = None

    if sc_code is not None:
        L1_code = n_samples ** 0.5 - (n_samples ** 0.5 - 1) * sc_code
        for i in range(0, n_components):
            tmp_code, _ = _projfunc(code[i, :][:, np.newaxis], L1_code, 1, nn=True)
            code[i, :] = tmp_code.reshape(-1)
    else:
        L1_code = None

    # initialize evaluation log

    if eval_log is True and true_dict is not None:
        time_lps, error, sparsity, atoms = \
            _evaluations(data, dictionary, code, true_dict=true_dict)
    else:
        time_lps, error, sparsity = \
            _evaluations(data, dictionary, code, true_dict=None)

    start_time = time_lps
    time_log = np.zeros(max_iter)
    error_log = np.zeros(max_iter)
    sparse_log = np.zeros(max_iter)

    if eval_log is True and true_dict is not None:
        atom_at_init = atoms
        previous_atom = atom_at_init

        atom_log = np.zeros(max_iter)

    objhistory = _error(data, dictionary, code)
    stepsize_dict = 1
    stepsize_code = 1
    for n_iter in range(1, max_iter + 1):

        if sc_code is not None:
            delta_code = np.dot(dictionary.T, (np.dot(dictionary, code) - data))
            begobj = objhistory

            while(True):
                code_new = code - stepsize_code * delta_code
                for i in range(0, n_components):
                    tmp_code, _ = _projfunc(code[i, :][:, np.newaxis], L1_code, 1, nn=True)
                    code_new[i, :] = tmp_code.reshape(-1)

                newobj = _error(data, dictionary, code_new)

                if newobj <= begobj:
                    break

                stepsize_code /= 2
                if stepsize_code < 1e-200:
                    print('Algorithm converged')
                    return dictionary, code, n_iter, logs

            stepsize_code *= 1.2
            code = code_new
        else:
            code = code * (np.dot(dictionary.T, data)) / \
            (np.dot(np.dot(dictionary.T, dictionary), code) + EPSILON)

            norms = np.sqrt(sum(dictionary.T ** 2))
            code /= (np.dot(norms[:, np.newaxis], np.ones((1, n_samples))))
            dictionary *= np.dot(np.ones((n_features, 1)), norms[np.newaxis, :])

        if sc_dict is not None:
            delta_dict = np.dot((np.dot(dictionary, code) - data), code.T)
            begobj = _error(data, dictionary, code)

            while(True):
                dict_new = dictionary - stepsize_dict * delta_dict
                norms = np.sqrt(sum(dict_new ** 2))
                for i in range(0, n_components):
                    tmp_dict, _ = _projfunc(new_dict[:, i][:, np.newaxis], L1_dict * norms[i],
                                            norms[i] ** 2, nn=True,
                                            verbose=verbose)
                    dict_new[:, i] = tmp_dict.reshape(-1)
                newobj = _error(data, dict_new, code)

                if newobj <= begobj:
                    break
                stepsize_dict /= 2

                if stepsize_dict < 1e-200:
                    print('Algorithm converged')
                    return dictionary, code, n_iter, logs

            stepsize_dict *= 1.2
            dictionary = dict_new

        else:
            dictionary = dictionary * (np.dot(data, code.T)) / \
            (np.dot(dictionary, np.dot(code, code.T)) + EPSILON)

        code_ = code.copy()
        code_[code_ < 0.0000001] = 0
        
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

        if n_iter % 10 == 0:

            error = np.linalg.norm(data - np.dot(dictionary, code), ord='fro')
            rel_error =  linnorm(data - np.dot(dictionary, code), 'fro') ** 2 / linnorm(data, 'fro') ** 2
            #rel_error =  linnorm(data - np.dot(dictionary, code), 'fro') ** 2 / linnorm(data, 'fro') ** 2
            sp = _hoyers_sparsity(normalize(code, axis=1))

            if verbose:
                iter_time = time.time()

                if true_dict is not None and eval_log:
                    print("Epoch %02d reached after %.3f seconds, error: %f, sparsity (mean): %f, atom: %f" %
                          (n_iter, iter_time - start_time, rel_error, np.mean(logged_sp), logged_atom))
                else:
                    print("Epoch %02d reached after %.3f seconds, error: %f, sparsity (mean): %f" %
                          (n_iter, iter_time - start_time, rel_error, np.mean(sp)))

    if eval_log is None:
        return dictionary, code, n_iter

    else:
        if eval_log is True and true_dict is not None:
            logs = {'time': time_log, 'error': error_log,
                    'sparsity': sparse_log, 'atoms': atom_log}
        else:
            logs = {'time': time_log, 'error': error_log,
                    'sparsity': sparse_log}

        return dictionary, code, n_iter, logs

def _nmf_sparse_constraint(data, n_components, true_dict=None,
                           solver='mu', sc_dict=None, sc_code=None,
                           max_iter=1000, tol=1e-4,
                           verbose=0, eval_log=None):
    if solver == 'mu':
        if eval_log is None:
            dictionary, code, n_iter = \
                _multiplicative_update(data, n_components,
                                       sc_dict=sc_dict, sc_code=sc_code,
                                       tol=tol,
                                       max_iter=max_iter, true_dict=true_dict,
                                       verbose=verbose, eval_log=eval_log)
            return dictionary, code, n_iter

        else:
            dictionary, code, n_iter, logs = \
                _multiplicative_update(data, n_components,
                                       sc_dict=sc_dict, sc_code=sc_code,
                                       tol=tol,
                                       max_iter=max_iter, true_dict=true_dict,
                                       verbose=verbose, eval_log=eval_log)
            return dictionary, code, n_iter, logs
    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)


class NMFsc(object):
    def __init__(self, n_components, tol=1e-4, solver='mu',
                 sc_dict=None, sc_code=None,
                 max_iter=1000, verbose=0, eval_log=None):
        """
        n_components
            number of components in dictionary
        tol:
            tolerance of convergence
        solver:
            ['mu'] 'mu': multiplicative update
        sc_dict:
            strength of sparseness for dictionary, in [0, 1]
        sc_code:
            strength of sparseness for code, in [0, 1]
        max_iter:
            maximum iteration
        verbose:
            verbose flag
        eval_log:
            flag of evaluation log
        """
        self.n_components = n_components
        self.tol = tol
        self.solver = solver
        self.sc_dict = sc_dict
        self.sc_code = sc_code
        self.max_iter = max_iter
        self.verbose = verbose
        self.eval_log = eval_log

    def fit_transform(self, data, true_dict=None, return_iter=None):

        if self.eval_log is None:

            dictionary, code, n_iter = \
                _nmf_sparse_constraint(data=data,
                                       n_components=self.n_components,
                                       tol=self.tol,
                                       true_dict=true_dict,
                                       solver=self.solver,
                                       sc_dict=self.sc_dict,
                                       sc_code=self.sc_code,
                                       max_iter=self.max_iter,
                                       verbose=self.verbose,
                                       eval_log=self.eval_log)
        else:
            dictionary, code, n_iter, logs = \
                _nmf_sparse_constraint(data=data,
                                       n_components=self.n_components,
                                       tol=self.tol,
                                       true_dict=true_dict,
                                       solver=self.solver,
                                       sc_dict=self.sc_dict,
                                       sc_code=self.sc_code,
                                       max_iter=self.max_iter,
                                       verbose=self.verbose,
                                       eval_log=self.eval_log)
            self.logs = logs

        self.dictionary = dictionary

        if return_iter:
            return code, n_iter
        else:
            return code