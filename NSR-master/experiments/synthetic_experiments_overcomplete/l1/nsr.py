import numpy as np
import math
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from numpy.linalg import norm as linnorm
import warnings
import time
from projfunc import projfunc

EPSILON = np.finfo(np.float32).eps


def _make_nn_sparse_coded_signal(n_samples, n_components, n_features,
                                 n_nonzero_coefs, random_state=None , alpha=0):
    """Generate a signal as a sparse combination of dictionary elements.
    Returns a matrix Y = DX, such as D is (n_features, n_components),
    X is (n_components, n_samples) and each column of X has exactly
    n_nonzero_coefs non-zero elements.
    Read more in the :ref:`User Guide <sample_generators>`.
    Parameters
    ----------
    n_samples : int
        number of samples to generate
    n_components :  int,
        number of components in the dictionary
    n_features : int
        number of features of the dataset to generate
    n_nonzero_coefs : int
        number of active (non-zero) coefficients in each sample
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    data : array of shape [n_features, n_samples]
        The encoded signal (Y).
    dictionary : array of shape [n_features, n_components]
        The dictionary with normalized components (D).
    code : array of shape [n_components, n_samples]
        The sparse code such that each column of this matrix has exactly
        n_nonzero_coefs non-zero items (X).
    """
    generator = check_random_state(random_state)
    if alpha < 0:
        raise ValueError("The input should be non-negative value"
                         "got %r" % (alpha))

    # generate dictionary
    D = abs(generator.randn(n_features, n_components))
    D /= np.sqrt(np.sum((D ** 2), axis=0))

    # generate code
    X = np.zeros((n_components, n_samples))
    for i in range(n_samples):
        idx = np.arange(n_components)
        generator.shuffle(idx)
        idx = idx[:n_nonzero_coefs]
        if alpha == 0:
            X[idx, i] = abs(np.random.randn(n_nonzero_coefs))
        else:
            X[idx, i] = np.array([max(d, 1) for d in abs(np.random.randn(n_nonzero_coefs)) + alpha])


    # encode signal
    Y = np.dot(D, X)

    return map(np.squeeze, (Y, D, X))


def _hoyers_sparsity(A):
    """
     Hoyer's sparsity
     Article: http://www.jmlr.org/papers/v5/hoyer04a.html

     This function takes value [0, 1].
      0 means the lowest sparsity and 1 means the most sparsity.
    """
    N = A.shape[1]
    sp = []

    for a in A:
        numerator_1 = np.sqrt(N) - (sum(abs(a.T)))
        numerator_2 = (np.sqrt(sum(a.T ** 2)) + EPSILON)
        numerator = numerator_1 / (numerator_2 + EPSILON)
        denominator = np.sqrt(N) - 1
        sp_a = min(1, max(0, abs(numerator / (denominator + EPSILON))))
        sp.append(sp_a)

    return sp

def _reorder_matrix(idx, A):
    perm_A = np.zeros(np.shape(A))
    for i in range(0, len(idx)):
        perm_A[idx[i], :] = A[i, :]
    return perm_A


def _support(x):
    
    idxs = np.where(np.array(x) > 1e-2)
    if len(x.shape) is 2:
        return set(zip(idxs[0], idxs[1]))
    elif len(x.shape) is 1:
        return set(*idxs)
    else:
        raise ValueError("The input should be 1-dim or 2-dim numpy.array"
                         "got %r-dim %r" % (len(x), x.__class__.__name__))
        return

def _support_dist(t_x, e_x):

    t_sup = _support(t_x)
    e_sup = _support(e_x)

    numerator = max(len(e_sup), len(t_sup)) - len(e_sup & t_sup)
    denominator = max(len(e_sup), len(t_sup))

    return numerator / denominator


def _l2_error(t_x, e_x):
    if len(t_x.shape) is 1 and len(e_x.shape) is 1:
        return linnorm(t_x - e_x) / (linnorm(t_x) ** 2)
    else:
        raise ValueError("The inputs should be 1-dimension numpy.array vectors"
                         " got %r-dim %r" % (len(t_x), t_x.__class__.__name__))
        return

def _calc_error(data, dictionary, code):
    return np.linalg.norm(data - np.dot(dictionary, code), ord='fro') / np.prod(data.shape)


def _calc_cost(data, dictionary, code, alpha, w):
    error = linnorm(data - np.dot(dictionary, code), ord='fro')
    sp = np.sum(np.sum(w * code))
    return error + alpha * sp


def _count_codes(_true_A, _est_A, axis=1, norm=True):

    num_recovered = 0
    num_codes = len(_true_A[0])

    # normalization
    if norm is True:
        est_A = normalize(_est_A, axis=axis)
        true_A = normalize(_true_A, axis=axis)
    else:
        est_A = _est_A
        true_A = _true_A

    # calculate ratio
    l2errors = np.array([_l2_error(t, e) for t, e in zip(true_A, est_A)])
    num_recovered = sum(l2errors[l2errors < 0.01])

    recovered_rate = 100 * (num_recovered / num_codes)
    return recovered_rate


def _count_atoms(_est_A, _true_A, axis=0, return_mat=None, norm=True, return_idx=None):
    """ Count recovered atoms
    Parameters
    ----------
    _est_A : array, shape(n_features, n_samples)
        estimated matrix
    _true_A : array, shape(n_features, n_samples)
        true matrix
    axis : 0 or 1, optional (1 by default)
        axis used to normalize the data along.
        If 1,            independently normalize each sample,
        otherwise (if 0) normalize each feature.
    """

    if not (_est_A.shape == _true_A.shape):
        raise ValueError("The shape of dictionaries should be same;"
                         "got %r and %r " % (_est_A.shape, _true_A.shape))
    num_recovered = 0
    num_atoms = len(_true_A[0])

    if norm is True:
        est_A = normalize(_est_A, axis=axis)
        true_A = normalize(_true_A, axis=axis)
    else:
        est_A = _est_A
        true_A = _true_A

    t_atoms = []
    e_atoms = []
    perm_idx = []

    for e in est_A.T:
        distances = [1 - np.abs(np.dot(e, t)) for t in true_A.T]
        min_idx = np.argmin(distances)
        min_t = true_A[:, min_idx]
        min_dis = distances[min_idx]

        perm_idx.append(min_idx)

        if (min_dis < 0.01):
            num_recovered += 1

            t_atoms.append(min_t)
            e_atoms.append(e)

    recovered_rate = 100 * (num_recovered / num_atoms)

    if return_mat:
        if return_idx:
            return recovered_rate, t_atoms, e_atoms, perm_idx
        else:
            return recovered_rate, t_atoms, e_atoms

    return recovered_rate


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
    n_features, n_components = dictionary.shape
    n_components, n_samples = code.shape
    norms = np.sqrt(sum(dictionary ** 2))
    norms = norms[:, np.newaxis].T
    dictionary = dictionary / np.dot(np.ones((n_features, 1)), norms)
    code = code * np.dot(norms.T, np.ones((1, n_samples)))

    t1 = 3
    t2 = 0.999
    error = sum((data - np.dot(dictionary, code)) ** 2)
    gram = np.dot(dictionary.T, dictionary)
    gram = gram - np.diag(np.diag(gram))

    for i in range(0, n_components):
        if (max(gram[i, :]) > t2) or (len(*np.nonzero(abs(code[i, :]) > 1e-7)) <= t1):
            val = np.max(error)
            pos = np.argmax(error)
            error[pos] = 0
            dictionary[:, i] = data[:, pos] / np.linalg.norm(data[:, pos])
            gram = np.dot(dictionary.T, dictionary)
            gram = gram - np.diag(np.diag(gram))

    return dictionary


def _multiplicative_update_code_l1(Y, H, U, alpha):
    numerator = H.T.dot(Y) - alpha
    denominator = H.T.dot(H).dot(U)
    delta_code = numerator / (denominator + EPSILON)
    # delta_code[delta_code < 1e-5] = EPSILON
    return delta_code


def _multiplicative_update_code_wl1(Y, H, U, alpha, p, e, return_w=None):

    n_components, n_samples = U.shape
    #W_denom = (abs(U) ** (1 - p)) + e
    W_denom = np.max(abs(U), 0) ** (1 - p) + e
    W = np.divide(p, W_denom)

    numerator = H.T.dot(Y) - alpha * 0.5 * W
    denominator = H.T.dot(H).dot(U)
    delta_code = numerator / (denominator + EPSILON)
    # delta_code[delta_code < 1e-5] = EPSILON

    if return_w:
        return delta_code, W
    else:
        return delta_code


def _multiplicative_update_dictionary(Y, H, U):
    numerator = Y.dot(U.T)
    denominator = H.dot(U.dot(U.T))

    # denominator[denominator == 0] = EPSILON
    delta_dictionary = numerator / (denominator + EPSILON)

    return delta_dictionary


def _renormarize(H, U):
    H = normalize(H)
    U = normalize(U)
    return H, U


def _evaluations(data, dictionary, code, alpha, weight, true_dict=None):
    _time = time.time()
    _error = _calc_error(data, dictionary, code)
    _cost = _calc_cost(data, dictionary, code, alpha, weight)
    _sparsity = _hoyers_sparsity(normalize(code, axis=1))

    if true_dict is not None:
        _atoms = _count_atoms(dictionary, true_dict)
        return _time, _error, _cost, _sparsity, _atoms
    else:
        return _time, _error, _cost, _sparsity


def _multiplicative_update(data, n_components, constraint='l1', p=None, e=0.01, max_iter=1000, alpha=0.,
                           true_dict=None, tol=1e-4, verbose=0, eval_log=None):

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

    for n_iter in range(1, max_iter + 1):

        dictionary = _clear_dictionary(dictionary, code, data)

        if constraint == 'l1':
            delta_code = \
                _multiplicative_update_code_l1(data, dictionary, code, alpha)
        elif constraint == 'wl1':
            if n_iter == 1:
                delta_code = \
                    _multiplicative_update_code_l1(data, dictionary, code, alpha)
            else:
                delta_code, weight = \
                    _multiplicative_update_code_wl1(data, dictionary, code, alpha, p, e, return_w=True)
        code *= delta_code

        delta_dictionary = \
            _multiplicative_update_dictionary(data, dictionary, code)
        dictionary *= delta_dictionary

        # for non-negativity
        isneg = np.where(code < 0, True, False)
        code = (-2 * isneg + 1) * code
        isneg = np.where(dictionary < 0, True, False)
        dictionary = (-2 * isneg + 1) * dictionary


        n_features, n_samples = data.shape
        norms = np.sqrt(sum(code.T ** 2))
        norms = norms[:, np.newaxis].T
        code = code / np.dot(norms.T, np.ones((1, n_samples)))
        dictionary = dictionary * np.dot(np.ones((n_features, 1)), norms)

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

        cost = _calc_cost(data, dictionary, code, alpha, weight)
        ref_tol = (previous_cost - cost) / previous_cost
        if ref_tol < tol:
            break

        # test convergence criterion every 10 iterations
        if n_iter % 10 == 0:

            error = np.linalg.norm(data - np.dot(dictionary, code), ord='fro')
            sp = _hoyers_sparsity(normalize(code, axis=1))

            if verbose:
                iter_time = time.time()

                if true_dict is not None and eval_log:
                    print("Epoch %02d reached after %.3f seconds, error: %f, sparsity (mean): %f, cost: %f, atom: %f" %
                          (n_iter, iter_time - start_time, error, np.mean(logged_sp), cost, logged_atom))
                else:
                    print("Epoch %02d reached after %.3f seconds, error: %f, sparsity (mean): %f, cost: %f" %
                          (n_iter, iter_time - start_time, error, np.mean(sp), cost))

            previous_cost = cost
    code[code < 1e-4] = 0

    # code = code * np.dot(norms.T, np.ones((1, n_samples)))

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
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
                                       solver='mu', constraint='l1', p=None, e=0.01,
                                       max_iter=1000,
                                       alpha=0., tol=1e-4,
                                       verbose=0, eval_log=None):
    if solver == 'mu':
        if eval_log is None:
            dictionary, code, n_iter = \
                _multiplicative_update(data, n_components,
                                       constraint=constraint, p=p, e=e,
                                       alpha=alpha, tol=tol,
                                       max_iter=max_iter, true_dict=true_dict,
                                       verbose=verbose, eval_log=eval_log)
            return dictionary, code, n_iter

        else:
            dictionary, code, n_iter, logs = \
                _multiplicative_update(data, n_components,
                                       constraint=constraint, p=p, e=e,
                                       alpha=alpha, tol=tol,
                                       max_iter=max_iter, true_dict=true_dict,
                                       verbose=verbose, eval_log=eval_log)
            return dictionary, code, n_iter, logs
    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)



class NSR(object):
    def __init__(self, n_components, alpha=0., tol=1e-4, solver='mu',
                 constraint='l1', p=None, e=0.01, max_iter=1000, verbose=0, eval_log=None):
        self.n_components = n_components
        self.alpha = alpha
        self.tol = tol
        self.solver = solver
        self.constraint = constraint
        self.p = p
        self.e = e
        self.max_iter = max_iter
        self.verbose = verbose
        self.eval_log = eval_log

    def fit_transform(self, data, true_dict=None, return_iter=None):

        if self.eval_log is None:

            dictionary, code, n_iter = \
                _nonnegative_sparse_representation(data=data,
                                                   n_components=self.n_components,
                                                   alpha=self.alpha,
                                                   tol=self.tol,
                                                   true_dict=true_dict,
                                                   solver=self.solver,
                                                   constraint=self.constraint,
                                                   p=self.p, e=self.e,
                                                   max_iter=self.max_iter,
                                                   verbose=self.verbose,
                                                   eval_log=self.eval_log)
        else:
            dictionary, code, n_iter, logs = \
                _nonnegative_sparse_representation(data=data,
                                                   n_components=self.n_components,
                                                   alpha=self.alpha,
                                                   tol=self.tol,
                                                   true_dict=true_dict,
                                                   solver=self.solver,
                                                   constraint=self.constraint,
                                                   p=self.p, e=self.e,
                                                   max_iter=self.max_iter,
                                                   verbose=self.verbose,
                                                   eval_log=self.eval_log)
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