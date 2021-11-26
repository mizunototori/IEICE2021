import numpy as np
import math
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from numpy.linalg import norm as linnorm
import warnings
import time
from nsr import _evaluations
from nsr import _norm_dict
from nsr import _norm_code
from nsr import _multiplicative_update_code_l1
from nsr import _multiplicative_update_code_wl1
from nsr import _initialize_nsr
from nsr import _calc_cost
from nsr_utils import _projfunc
from nsr_utils import _hoyers_sparsity
from nsr_utils import _support
from nsr_utils import _support_dist
from nsr_utils import _l2_error
from nsr_utils import _calc_error
from nsr_utils import _count_codes
from nsr_utils import _get_psnr

def _mm_update(data, dictionary, true_code, constraint='l1', nn_method='rect', p=None,
               e=0.01, max_iter=10000, alpha=0., tol=1e-4,
               verbose=0, eval_log=None):

    if (constraint is 'wl1') and ((p is None) or not(0 < p < 1)):
        raise ValueError('Invalid p parameter: got %r instead of a float (0, 1). ' % p)
    _data = data.copy()
    #data = normalize(data, axis=0)
    n_components = np.shape(dictionary)[1]
    _, code = _initialize_nsr(data, n_components)
    weight = np.ones(code.shape)

    time_lps, error, cost, sparsity = _evaluations(_data, dictionary, code, alpha, weight, true_dict=None)

    start_time = time_lps
    cost_at_init = cost
    previous_cost = cost_at_init

    time_log = np.zeros(max_iter)
    error_log = np.zeros(max_iter)
    sparse_log = np.zeros(max_iter)
    cost_log = np.zeros(max_iter)
    psnr_log = np.zeros(max_iter)


    alpha = 0.8 * max(abs(code))
    p = 0.8 * max(abs(code))

    for n_iter in range(1, max_iter + 1):
        #code = _norm_code(code)

        if constraint == 'l1':
            alpha = alpha * 0.8

            delta_code = \
                _multiplicative_update_code_l1(data, dictionary, code, alpha)
        elif constraint == 'wl1':
            if n_iter == 1:
                delta_code = \
                    _multiplicative_update_code_l1(data, dictionary, code, alpha)
            else:
                p = p * 0.99
                alpha = alpha * 0.8
                delta_code, weight = \
                    _multiplicative_update_code_wl1(data, dictionary, code, alpha, p, e, return_w=True)

        old_code = code.copy()
        code *= delta_code

        # for non-negativity
        n_samples = np.shape(code)[1]
        n_components = np.shape(dictionary)[1]
        if nn_method == 'projfunc':
            nn = 1
            #sp_code = 0.0001
            #k1 = max(np.sqrt(n_samples)-(np.sqrt(n_samples)-1) * sp_code, 1.0)
            #k2 = 1
            for i in range(1, n_samples):
                k1 = max(sum(abs(code[:, i])), 1.0)
                k2 = np.sqrt(sum(code[:, i] ** 2)) ** 2
                code[:, i] = _projfunc(code[:, i, np.newaxis], k1, k2, nn, verbose=False)[0].reshape(n_components)
        elif nn_method == 'rect':
            code[code < 0] = 0
            dictionary[dictionary < 0] = 0
        else:
            code[code < 0] = 0
            dictionary[dictionary < 0] = 0

        cost = _calc_cost(data, dictionary, code, alpha, weight)
        ref_tol = (previous_cost - cost) / previous_cost
        previous_cost = cost
        if abs(ref_tol) < tol:
           break

        # test convergence criterion every 10 iterations
        if n_iter % 1 == 0:
            if eval_log:
                time_lps, error, cost, sparsity = _evaluations(_data, dictionary, code, alpha, weight, true_dict=None)

                logged_error = error
                logged_sp = sparsity
                logged_cost = cost
                iter_time = time_lps
                logged_time = iter_time - start_time
                psnr = _get_psnr(true_code, code)

                error_log[n_iter - 1] = logged_error
                sparse_log[n_iter - 1] = np.mean(logged_sp)
                time_log[n_iter - 1] = logged_time
                cost_log[n_iter - 1] = logged_cost
                psnr_log[n_iter - 1] = psnr

            error = np.linalg.norm(data - np.dot(dictionary, code), ord='fro')
            sp = _hoyers_sparsity(normalize(code, axis=0))

            if verbose:
                iter_time = time.time()

                print("Epoch %02d reached after %.3f seconds, error: %f, sparsity (mean): %f, cost: %f, psnr: %f"%
                      (n_iter, iter_time - start_time, error, np.mean(sp), cost, psnr))


    # code = code * np.dot(norms.T, np.ones((1, n_samples)))

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 1 != 0):
        end_time = time.time()
        print("Epoch %02d reached after %.3f seconds." %
              (n_iter, end_time - start_time))

    if eval_log is None:
        return dictionary, code, n_iter

    else:
        logs = {'time': time_log, 'error': error_log, 'cost': cost_log,
                'sparsity': sparse_log, 'psnr': psnr_log}

        return code, n_iter, logs

def _nonnegative_sparse_coding(data, dictionary, true_code,
                               solver='mu', nn_method='rect',
                               constraint='l1', p=None, e=0.01,
                               max_iter=1000,
                               alpha=0., tol=1e-4,
                               verbose=0, eval_log=None):
    if solver == 'mu':
        if eval_log is None:
            code, n_iter = \
                _mm_update(data, dictionary, true_code,
                           constraint=constraint, p=p, e=e,
                           alpha=alpha, tol=tol, nn_method=nn_method,
                           max_iter=max_iter,
                           verbose=verbose, eval_log=eval_log)
            return code, n_iter

        else:
            code, n_iter, logs = \
                _mm_update(data, dictionary, true_code,
                           constraint=constraint, p=p, e=e,
                           alpha=alpha, tol=tol, nn_method=nn_method,
                           max_iter=max_iter,
                           verbose=verbose, eval_log=eval_log)
            return code, n_iter, logs
    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)


class NSC(object):
    def __init__(self, alpha=0., tol=1e-4, solver='mu', nn_method='rect',
                 constraint='l1', p=None, e=0.01, max_iter=1000, verbose=0,
                 eval_log=None):
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

    def fit_transform(self, data, dictionary, true_code, return_iter=None):

        if self.eval_log is None:

            code, n_iter = \
                _nonnegative_sparse_coding(data=data,
                                           dictionary=dictionary,
                                           true_code=true_code,
                                           alpha=self.alpha,
                                           tol=self.tol,
                                           solver=self.solver,
                                           nn_method=self.nn_method,
                                           constraint=self.constraint,
                                           p=self.p, e=self.e,
                                           max_iter=self.max_iter,
                                           verbose=self.verbose,
                                           eval_log=self.eval_log)
        else:
            code, n_iter, logs = \
                _nonnegative_sparse_coding(data=data,
                                           dictionary=dictionary,
                                           true_code=true_code,
                                           alpha=self.alpha,
                                           tol=self.tol,
                                           solver=self.solver,
                                           nn_method=self.nn_method,
                                           constraint=self.constraint,
                                           p=self.p, e=self.e,
                                           max_iter=self.max_iter,
                                           verbose=self.verbose,
                                           eval_log=self.eval_log)
            self.logs = logs


        if return_iter:
            return code, n_iter
        else:
            return code
