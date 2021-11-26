''' Test code for nsr_utils._hoyers_sparsity
Test environment: pytest
'''

import numpy as np
from numpy.matlib import repmat
from nsr_utils import _projfunc
from nsr_utils import _hoyers_sparsity
from nsr import _make_nn_sparse_coded_signal
from sklearn.preprocessing import normalize
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_almost_equal

def test_projfunc_sparse_vector():
    # Dimension
    K = 5

    # Desired sparsity
    ds = 0.8

    # Input vector
    x = np.random.rand(K)

    # Generate sparse vector
    k1 = np.sqrt(K) - (np.sqrt(K)-1) * ds
    x = _projfunc(x[:, np.newaxis], k1, 1, nn=1)[0].reshape(K)

    # Measure sparsity
    sp = _hoyers_sparsity(x)

    print(sp, x.shape)
    assert_almost_equal(sp, ds, decimal=4)

def test_projfunc_sparse_matrix():
    # The dimension of matrix is KxN
    K = 5
    N = 10

    # Desired sparsity
    ds = 0.8
    dss = repmat(ds, 1, N).reshape(N)

    # Input vector
    x = np.random.rand(K, 1)

    # Generate sparse vector
    k1 = np.sqrt(K) - (np.sqrt(K)-1) * ds
    x = _projfunc(x, k1, 1, nn=1)[0]

    # Generate column-sparse matrix
    X = repmat(x, 1, N)
    # Measure sparsity
    sp = _hoyers_sparsity(X)
    print(sp)

    assert_almost_equal(sp, dss, decimal=4)

def test_make_sparse_signal_low_sparse_matrix():
    # The dimension of matrix is KxN
    n_features = 5
    n_components = 10
    n_samples = 20
    n_nonzero_coefs = 5
    true_data, true_dictionary, true_code = \
        _make_nn_sparse_coded_signal(n_samples=n_samples,
                                     n_components=n_components,
                                     n_features=n_features,
                                     n_nonzero_coefs=n_nonzero_coefs,
                                     random_state=0)

    # Desired sparsity
    ds = 0.25

    # Measure sparsity
    sp = _hoyers_sparsity(true_code)
    print(np.mean(sp))

    assert_true(abs(ds - np.mean(sp) < 0.2))


def test_make_sparse_signal_high_sparse_matrix():
    # The dimension of matrix is KxN
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

    # Desired sparsity
    ds = 0.81

    # Measure sparsity
    sp = _hoyers_sparsity(normalize(true_code, axis=0))
    print(np.mean(sp))

    assert_true(np.mean(sp) == ds)
#    assert_true(abs(ds - np.mean(sp) < 0.2))

