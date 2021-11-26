import numpy as np
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from numpy.linalg import norm as linnorm
import cmath
import matplotlib.pyplot as plt

EPSILON = np.finfo(np.float32).eps


def _get_psnr(im, recon):
    """ PSNRを得る """ 
    return 10. * np.log(im.max() / np.sqrt(np.mean((im - recon) ** 2)))


def _show_atoms(_est_A, _true_A, axis=0, return_mat=None,
                 norm=True, return_idx=None):
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

            t_atoms.append(min_t)
            e_atoms.append(e)

        for i in range(1, len(t_atoms)):
            plt.subplot(len(t_atoms), 1, i)
            plt.plot(t_atoms[i])
            plt.plot(e_atoms[i])
def _make_nn_sparse_coded_signal(n_samples, n_components, n_features,
                                 n_nonzero_coefs, random_state=None, alpha=0):
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
            X[idx, i] = np.array([max(d, 1) for d in abs(np.random.randn(n_nonzero_coefs) * 4) + alpha])

    # encode signal
    Y = np.dot(D, X)

    return Y, D, X
    #return map(np.squeeze, (Y, D, X))


def _hoyers_sparsity(A):
    """
     Hoyer's sparsity
     Article: http://www.jmlr.org/papers/v5/hoyer04a.html

     This function takes value [0, 1].
      0 means the lowest sparsity and 1 means the most sparsity.
    """
    def vector_sp(A):
        N = len(A)
        numerator_1 = np.sqrt(N) - (sum(abs(A.T)))
        numerator_2 = (np.sqrt(sum(A.T ** 2)) + EPSILON)
        numerator = numerator_1 / (numerator_2 + EPSILON)
        denominator = np.sqrt(N) - 1
        sp = min(1, max(0, abs(numerator / (denominator + EPSILON))))
        return sp

    sp = []
    #N = A.shape[1]
    if len(A.shape) == 1:
        return vector_sp(A)

    else:
        sp = []
        for a in A.T:
            sp.append(vector_sp(a))
        return sp


def _reorder_matrix(idx, A):
    perm_A = np.zeros(np.shape(A))
    for i in range(0, len(idx)):
        perm_A[idx[i], :] = A[i, :]
    return perm_A


def _support(x, eps=1e-7):
    idxs = np.where(np.array(x) > eps)
    if len(x.shape) is 2:
        return set(zip(idxs[0], idxs[1]))
    elif len(x.shape) is 1:
        return set(*idxs)
    else:
        raise ValueError("The input should be 1-dim or 2-dim numpy.array"
                         "got %r-dim %r" % (len(x), x.__class__.__name__))
        return


def _support_dist(t_x, e_x, eps=1e-7):

    t_sup = _support(t_x, eps=eps)
    e_sup = _support(e_x, eps=eps)

    numerator = max(len(e_sup), len(t_sup)) - len(e_sup & t_sup)
    denominator = max(len(e_sup), len(t_sup))

    return numerator / denominator


def _l2_error(t_x, e_x):
    return linnorm(t_x - e_x) / (linnorm(t_x) ** 2 + EPSILON)
    '''
    else:
        raise ValueError("The inputs should be 1-dimension numpy.array vectors"
                         " got %r-dim %r" % (len(t_x), t_x.__class__.__name__))
        return
    '''

def _calc_error(data, dictionary, code):
    error = np.linalg.norm(data - np.dot(dictionary, code), ord='fro')
    return error / np.prod(data.shape)


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


def _count_atoms(_est_A, _true_A, axis=0, return_mat=None,
                 norm=True, return_idx=None):
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


# This code rewrite hoyer's nmf matlab code.
# Ref: http://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf
import numpy as np
import copy
import pickle
EPSILON = np.finfo(np.float32).eps


def _projfunc(s, k1, k2, nn=True, verbose=False):

    """Soves the following problem:
    Given a vector s, find the vector v having sum(abs(v)) = k1
    and sum(v.^2)= k2 which is closest to s  in the euclidean sense.
    If the vinary flag nn is set the vector v is additionally
    restricted to being non-negative (v >= 0)

    Parameters:
    -----------
    s: given vector
    k1: L1 constraint
    k2: L2 constraint
    nn: nonnegative flag
    verbose: it prints progress of v
    """
    if k1 < 1.0:
        raise ValueError("k1 should greater than 1.0; but k1 was given %r." % k1)

    s = np.array(s)
    # Problem dimension
    N = len(s)
    if s.shape[0] != N:
        raise ValueError("The shape of given vector s should be (%r, %r), but given %r." % (N, 1, s.shape))

    # If non-negativity flag not set, record signs and take abs
    if not nn:
        isneg = s < 0
        isneg = np.where(s < 0, 1, 0)
        s = np.abs(s)

    # Start by projecting the point to the sum constraint hyperplane
    v = s + (k1 - np.sum(s)) / N

    # Initialize zerocoeff (initially, no elements are assumed zero)
    zerocoeff = []

    j = 0
    while(1):

        # This does the proposed projection operator

        midpoint = np.ones((N, 1)) * k1 / (N - len(zerocoeff))
        midpoint.flat[zerocoeff] = 0

        # if np.all(v == midpoint):
        #    raise ValueError('v and midpoint is same.')
        w = v - midpoint

        a = np.sum(w ** 2)
        b = np.dot(2 * w.T, v)
        c = np.sum(v ** 2) - k2
        alphap = (-b + np.real(cmath.sqrt(np.abs(b ** 2 - 4 * a * c)))) / (2 * a + EPSILON)
        if np.isnan(alphap):
            raise ValueError('v contains nan (0)')
        v_tmp = v
        v = alphap * w + v
        if np.any(np.isnan(v)):
            raise ValueError('v contains nan (1).')

        if np.all(v >= 0):
            # We've found our solution
            usediters = j + 1
            break
        j = j + 1
        # Set negs to zero, subtract appropriate amount from rest
        zerocoeff = np.where(v <= 0)[0]
        v.flat[zerocoeff] = 0

        tempsum = np.sum(v)
        if N == len(zerocoeff):
            v = v + (k1 - tempsum) / (N - len(zerocoeff) + EPSILON)
        else:
            v = v + (k1 - tempsum) / (N - len(zerocoeff))
        v.flat[zerocoeff] = 0

        if np.any(np.isnan(v)):
            raise ValueError('v contains nan (2).')

    # If non-negativity flag not set, return signs to solution
    if not nn:
        v = (-2 * isneg + 1) * v

    # Check for problems
    if np.max(np.max(np.abs(np.imag(v)))) > 1e-10:
        raise ValueError('Somehow got imaginary values!')

    return v, usediters
