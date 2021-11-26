import numpy as np
import math
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from numpy.linalg import norm as linnorm
import warnings
import time
from nsr_utils import _projfunc
from nsr_utils import _make_nn_sparse_coded_signal
from nsr_utils import _hoyers_sparsity
from nsr_utils import _reorder_matrix
from nsr_utils import _support
from nsr_utils import _support_dist
from nsr_utils import _l2_error
from nsr_utils import _calc_error
from nsr_utils import _count_codes
from nsr_utils import _count_atoms
import cupy as cp


EPSILON = np.finfo(np.float32).eps

def _norm_dict(dic):
    xp = cp.get_array_module(dic)
    norms      = xp.sqrt(xp.sum(dic ** 2,axis=0))
    norms[norms==0]=1
    norms      = xp.expand_dims(norms, axis=1)
    return dic/norms.T

def _norm_code(code):
    xp = cp.get_array_module(code)
    norms = xp.sqrt(np.sum(code ** 2, axis=1))
    norms[norms==0]=1
    norms = xp.expand_dims(norms, axis=1)
    return code/norms

def _calc_cost(data, dictionary, code, alpha, w):
    xp = cp.get_array_module(data)
    error = xp.linalg.norm(data - xp.dot(dictionary, code), ord='fro')
    sp = xp.sum(w * code)
    return error + alpha * sp


def _initialize_nsr(Y, n_components):
    n_features, n_samples = Y.shape

    if not (n_features < n_components):
        warnings.warn("NSR expect over-complete dictionary; "
                      "The dimensions should be "
                      "n_features: %r < n_components: %r"
                      % (n_features, n_components))

    D = Y[:, :n_components].copy()
    X = np.random.ranf((n_components, n_samples))

    return D, X


def _clear_dictionary(dictionary, code, data):
    xp = cp.get_array_module(code)
    n_features, n_components = dictionary.shape
    n_components, n_samples = code.shape
    def _cov(dictionary):
        covaliance = xp.dot(dictionary.T, dictionary)
        return covaliance-xp.diag(xp.diag(covaliance))
    def _elim(dictionary, gram, data, error):
        for i in range(0, n_components):
            if (xp.max(gram[i, :]) > t2) or (xp.count_nonzero(np.abs(code[i, :]) > 1e-7) <= t1):
                val = xp.max(error)
                pos = xp.argmax(error)
                error[pos] = 0
                dictionary[:, i] = data[:, pos] / xp.linalg.norm(data[:, pos])
                gram  = _cov(dictionary)
        return dictionary
    def _err(dictionary, code, data):
        return xp.sum((data - xp.dot(dictionary, code)) ** 2,axis=0)

    t1 = 3
    t2 = 0.999
    error      = _err(dictionary, code, data)
    gram       = _cov(dictionary)
    dictionary = _elim(dictionary,gram,data,error)
    return dictionary


def _multiplicative_update_code_l1(Y, H, U, alpha):
    numerator = H.T.dot(Y) - alpha
    denominator = H.T.dot(H).dot(U)
    delta_code = numerator / (denominator + EPSILON)
    return delta_code


def _multiplicative_update_code_wl1(Y, H, U, alpha, p, e, return_w=None):
    W_denom = U ** (1 - p) + e
    W = np.divide(p, W_denom)
    numerator = H.T.dot(Y) - alpha * W
    denominator = H.T.dot(H).dot(U)
    delta_code = numerator / (denominator + EPSILON)
    if return_w:
        return delta_code, W
    else:
        return delta_code


def _multiplicative_update_dictionary(Y, H, U):
    numerator = Y.dot(U.T)
    denominator = H.dot(U.dot(U.T))
    delta_dictionary = numerator / (denominator + EPSILON)
    return delta_dictionary


def _evaluations(data, dictionary, code, alpha, weight, true_dict=None):
    _time = time.time()
    _error = _calc_error(data, dictionary, code)
    _cost = _calc_cost(data, dictionary, code, alpha, weight)
    _sparsity = _hoyers_sparsity(normalize(code, axis=0))
    if true_dict is not None:
        _atoms = _count_atoms(dictionary, true_dict)
        return _time, _error, _cost, _sparsity, _atoms
    else:
        return _time, _error, _cost, _sparsity


def _multiplicative_update(data, n_components, constraint='l1', nn_method='rect', p=1.0, e=0.01, max_iter=1000, alpha=0.,
                           true_dict=None, tol=1e-4, verbose=0, eval_log=None, gpu=False):

    if (constraint is 'wl1') and ((p is None) or not(0 < p < 1)):
        raise ValueError('Invalid p parameter: got %r instead of a float (0, 1). ' % p)
    _data = data.copy()
    data = normalize(data, axis=0)
    dictionary, code = _initialize_nsr(data, n_components)
    weight = np.ones(code.shape)

    if eval_log is True and true_dict is not None:
        time_lps, error, cost, sparsity, atoms = _evaluations(_data, dictionary, code, alpha, weight, true_dict=true_dict)
    else:
        time_lps, error, cost, sparsity = _evaluations(_data, dictionary, code, alpha, weight, true_dict=None)

    start_time = time_lps
    cost_at_init = cost
    previous_cost = cost_at_init

    time_log = np.zeros(max_iter)
    error_log = np.zeros(max_iter)
    sparse_log = np.zeros(max_iter)
    cost_log = np.zeros(max_iter)

    if eval_log is True and true_dict is not None:
        atom_at_init = atoms
        previous_atom = atom_at_init
        atom_log = np.zeros(max_iter)
    
    if p is None:
        p=1.0

    if(gpu):
        dict_xp = cp.asarray(dictionary)
        code_xp = cp.asarray(code)
        data_xp = cp.asarray(data)
        weight_xp = cp.asarray(weight)
        alpha_xp = cp.asarray(alpha)
        p_xp = cp.asarray(p)
        e_xp = cp.asarray(e)
    else:
        dict_xp = dictionary
        code_xp = code
        data_xp = data
        weight_xp = weight
        alpha_xp = alpha
        p_xp = p
        e_xp = e

    #dict_xp, code_xp = _norm_dict(dict_xp)
    for n_iter in range(1, max_iter + 1):
        dict_xp = _norm_dict(dict_xp)
        code_xp = _norm_code(code_xp)
        dict_xp = _clear_dictionary(dict_xp, code_xp, data_xp)

        if constraint == 'l1':
            delta_code = \
                _multiplicative_update_code_l1(data_xp, dict_xp, code_xp, alpha_xp)
        elif constraint == 'wl1':
            if n_iter == 1:
                delta_code = \
                    _multiplicative_update_code_l1(data_xp, dict_xp, code_xp, alpha_xp)
            else:
                delta_code, weight_xp = \
                    _multiplicative_update_code_wl1(data_xp, dict_xp, code_xp, alpha_xp, p_xp, e_xp, return_w=True)
        code_xp *= delta_code


        delta_dictionary = \
            _multiplicative_update_dictionary(data_xp, dict_xp, code_xp)
        dict_xp *= delta_dictionary

        # for non-negativity
        code_xp[code_xp < 0] = 0
        dict_xp[dict_xp < 0] = 0
        #assert (dict_xp>=0).all(), 'Some dictionary elements are negative. '
  
        cost = _calc_cost(data_xp, dict_xp, code_xp, alpha, weight_xp)
        ref_tol = (previous_cost - cost) / previous_cost
        previous_cost = cost
        if abs(ref_tol) < tol:
            break

        # test convergence criterion every 10 iterations
        if n_iter % 500 == 0:
            if(gpu):
                dictionary = cp.asnumpy(dict_xp)
                code       = cp.asnumpy(code_xp)
                data       = cp.asnumpy(data_xp)
                weight     = cp.asnumpy(weight_xp)
            else:
                dictionary = dict_xp
                code       = code_xp
                data       = data_xp
                weight     = weight_xp

            if eval_log:
                if true_dict is not None:
                    time_lps, error, cost, sparsity, atoms = _evaluations(_data, dictionary, code, alpha, weight, true_dict=true_dict)
                else: 
                    time_lps, error, cost, sparsity = _evaluations(_data, dictionary, code, alpha, weight, true_dict=None)


                logged_error = error
                logged_sp = sparsity
                logged_cost = cost
                iter_time = time_lps
                logged_time = iter_time - start_time

                error_log[n_iter - 1] = logged_error
                sparse_log[n_iter - 1] = np.mean(logged_sp)
                time_log[n_iter - 1] = logged_time
                cost_log[n_iter - 1] = logged_cost

                if true_dict is not None:
                    logged_atom = atoms
                    atom_log[n_iter - 1] = logged_atom

            error = np.linalg.norm(data - np.dot(dictionary, code), ord='fro')
            sp = _hoyers_sparsity(normalize(code, axis=0))

            if verbose:
                iter_time = time.time()
                if true_dict is not None and eval_log:
                    print("Epoch %02d reached after %.3f seconds, error: %f, sparsity (mean): %f, cost: %f, atom: %f" %
                          (n_iter, iter_time - start_time, error, np.mean(logged_sp), cost, logged_atom))
                else:
                    print("Epoch %02d reached after %.3f seconds, error: %f, sparsity (mean): %f, cost: %f" %
                          (n_iter, iter_time - start_time, error, np.mean(sp), cost))




    # code = code * np.dot(norms.T, np.ones((1, n_samples)))

    if(gpu):
        dictionary = cp.asnumpy(dict_xp)
        code       = cp.asnumpy(code_xp)
        data       = cp.asnumpy(data_xp)
        weight     = cp.asnumpy(weight_xp)
    else:
        dictionary = dict_xp
        code       = code_xp
        data       = data_xp
        weight     = weight_xp

    if eval_log:
        if true_dict is not None:
            time_lps, error, cost, sparsity, atoms = _evaluations(_data, dictionary, code, alpha, weight, true_dict=true_dict)
        else: 
            time_lps, error, cost, sparsity = _evaluations(_data, dictionary, code, alpha, weight, true_dict=None)


        logged_error = error
        logged_sp = sparsity
        logged_cost = cost
        iter_time = time_lps
        logged_time = iter_time - start_time

        error_log[n_iter - 1] = logged_error
        sparse_log[n_iter - 1] = np.mean(logged_sp)
        time_log[n_iter - 1] = logged_time
        cost_log[n_iter - 1] = logged_cost

        if true_dict is not None:
            logged_atom = atoms
            atom_log[n_iter - 1] = logged_atom

    # do not print if we have already printed in the convergence test
    if verbose :
        end_time = time.time()
        if true_dict is not None and eval_log:
            print("Epoch %02d reached after %.3f seconds. Atom recovery is %f." %
              (n_iter, end_time - start_time, atoms))
        else:
            print("Epoch %02d reached after %.3f seconds." %
              (n_iter, end_time - start_time))



    if eval_log is None:
        return dictionary, code, n_iter

    else:
        if  eval_log is True and true_dict is not None:
            logs = {'time': time_log, 'error': error_log, 'cost': cost_log,
                    'sparsity': sparse_log, 'atoms': atom_log}
        else:
            logs = {'time': time_log, 'error': error_log, 'cost': cost_log,
                    'sparsity': sparse_log}

        return dictionary, code, n_iter, logs





def _nonnegative_sparse_representation(data, n_components, true_dict=None,
                                       solver='mu', nn_method='rect',
                                       constraint='l1', p=None, e=0.01,
                                       max_iter=1000,
                                       alpha=0., tol=1e-4,
                                       verbose=0, eval_log=None,gpu=False):
    if solver == 'mu':
        if eval_log is None:
            dictionary, code, n_iter = \
                _multiplicative_update(data, n_components,
                                       constraint=constraint, p=p, e=e,
                                       alpha=alpha, tol=tol, nn_method=nn_method,
                                       max_iter=max_iter, true_dict=true_dict,
                                       verbose=verbose, eval_log=eval_log,gpu=gpu)
            return dictionary, code, n_iter

        else:
            dictionary, code, n_iter, logs = \
                _multiplicative_update(data, n_components,
                                       constraint=constraint, p=p, e=e,
                                       alpha=alpha, tol=tol, nn_method=nn_method,
                                       max_iter=max_iter, true_dict=true_dict,
                                       verbose=verbose, eval_log=eval_log,gpu=gpu)
            return dictionary, code, n_iter, logs
    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)



class NSR(object):
    def __init__(self, n_components, alpha=0., tol=1e-4, solver='mu', nn_method='rect',
                 constraint='l1', p=None, e=0.01, max_iter=1000, verbose=0, eval_log=None,gpu=False):
        self.n_components = n_components
        self.alpha = alpha
        self.tol = tol
        self.solver = solver
        self.nn_method = nn_method
        self.constraint = constraint
        self.p = p
        self.e = e
        self.max_iter = max_iter
        self.verbose = verbose
        self.eval_log = eval_log
        self.gpu=gpu

    def fit_transform(self, data, true_dict=None, return_iter=None):

        if self.eval_log is None:

            dictionary, code, n_iter = \
                _nonnegative_sparse_representation(data=data,
                                                   n_components=self.n_components,
                                                   alpha=self.alpha,
                                                   tol=self.tol,
                                                   true_dict=true_dict,
                                                   solver=self.solver,
                                                   nn_method=self.nn_method,
                                                   constraint=self.constraint,
                                                   p=self.p, e=self.e,
                                                   max_iter=self.max_iter,
                                                   verbose=self.verbose,
                                                   eval_log=self.eval_log,
                                                   gpu=self.gpu)
        else:
            dictionary, code, n_iter, logs = \
                _nonnegative_sparse_representation(data=data,
                                                   n_components=self.n_components,
                                                   alpha=self.alpha,
                                                   tol=self.tol,
                                                   true_dict=true_dict,
                                                   solver=self.solver,
                                                   nn_method=self.nn_method,
                                                   constraint=self.constraint,
                                                   p=self.p, e=self.e,
                                                   max_iter=self.max_iter,
                                                   verbose=self.verbose,
                                                   eval_log=self.eval_log,
                                                   gpu=self.gpu)
            self.logs = logs

        self.dictionary = dictionary

        if return_iter:
            return code, n_iter
        else:
            return code
    '''
    def fit_transform_logged(self, data, true_dict=None):

        self.dictionary = dictionary
        return code, logs
    '''