import numpy as numpy
import math


def _make_nn_sparse_coded_signal(n_samples, n_components, n_features,
                                 nonzero_coefs, random_state=None):
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

    # generate dictionary
    D = generator.randn(n_features, n_components)
    D /= np.sqrt(np.sum((D ** 2), axis=0))
    D = abs(D)

    # generate code
    X = np.zeros((n_components, n_samples))
    for i in range(n_samples):
        idx = np.arange(n_components)
        generator.shuffle(idx)
        idx = idx[:n_nonzero_coefs]
        X[idx, i] = generator.randn(n_nonzero_coefs)
    X = abs(X)

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
        numerator = numerator_1 / numerator_2
        denominator = np.sqrt(N) - 1
        sp_a = numerator / (denominator + EPSILON)
        sp.append(sp_a)

    return sp


def _count_atoms(_est_A, _true_A, axis=0):
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

    est_A = normalize(_est_A, axis=axis)
    true_A = normalize(_true_A, axis=axis)

    for e in est_A.T:

        distances = [np.mean((e - t) ** 2) for t in true_A.T]

        min_idx = np.argmin(distances)
        min_t = true_A[:, min_idx]

        # Assume normalization for atoms (Aatom' * Eatom = 1)
        dis_t = 1 - abs(np.dot(min_t.T, e))

        if (dis_t < 0.01):
            num_recovered += 1

    recovered_rate = 100 * (num_recovered / num_atoms)

    # print("recovered rate: %r" % recovered_rate)
    return recovered_rate


def _initialize_nsr(Y, n_components):
    n_features, n_samples = Y.shape
    H = Y[:, :n_components].copy()
    U = np.random.ranf(n_components * n_samples)
    U = U.reshape(n_components, n_samples)

    return H, U


def _multiplicative_update_code(Y, H, U):
    return H.T.dot(Y) / H.T.dot(H).dot(U)


def _multiplicative_update_dictionary(Y, H, U):
    return Y.dot(U.T) / H.dot(U.dot(U.T))


def _multiplicative_update(data, n_components, max_iter=1000,
                           true_dict=None, eval_log=True):

    if eval_log is True:
        error_history = []
        if true_dict is not None:
            atom_history = []

    dictionary, code = _initialize_nsr(data, n_components)

    for n_iter in range(1, max_iter + 1):

        delta_code =
        _multiplicative_update_code(data, dictionary, code)
        code *= delta_code

        delta_dictionary =
        _multiplicative_update_dictionary(data, dictionary, code)
        dictionary *= delta_dictionary

        if eval_log:
            error = sum(sum(pow(data - np.dot(dictionary, code), 2)))
            error_history.append(error)
            if true_dict is not None:
                atoms = _count_atoms(dictionary, true_dict)
                atom_history.append(atoms)

    return dictionary, code


def _nonnegative_sparse_representation(data, n_components, true_dict=None,
                                       solver='mu', eval_log=True):
    if solver == 'mu':
        dictionary, code = _multiplicative_update(data, n_components,
                                                  true_dict, eval_log)
    return dictionary, code


class NSR(object):
    def __init__(self, data, n_components, solver='mu', verbose=0,
                 true_dict=None):
        self.data = data
        self.n_components = n_components
        self.solver = solver
        self.verbose = verbose
        self.true_dict = true_dict

    def fit_transform(self, data, true_dict=None, solver='mu', eval_log=True,
                      verbose=0):

        dictionary, code =
        _nonnegative_sparse_representation(data=data,
                                           n_components=self.n_components,
                                           true_dict=self.true_dict,
                                           solver=self.solver,
                                           eval_log=eval_log)

        self.dictionary = dictionary
        
        return code

    '''
    def fit_transform_logged(self, data, true_dict=None):

        self.dictionary = dictionary
        return code, logs
    '''