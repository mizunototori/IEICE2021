#------------------------------------------------------------
# Testing for nsr_utils.py
#   - Test environment: pytest
#   - Modules:
#       - _make_nn_sparse_signal
#       - _projfunc
#       - _hoyers_sparsity
#------------------------------------------------------------

import numpy as np
from numpy.matlib import repmat
from nsr import _make_nn_sparse_coded_signal
from sklearn.preprocessing import normalize
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_almost_equal
from nsr_utils import _projfunc
from nsr_utils import _hoyers_sparsity
#------------------------------------------------------------
# Test for _make_nn_sparse_signals
#------------------------------------------------------------
def test_num_of_nonzero():

    n_features = 3
    n_components = 5
    n_samples = 10
    n_nonzero_coefs = 1
    true_data, true_dictionary, true_code = \
        _make_nn_sparse_coded_signal(n_samples=n_samples,
                                     n_components=n_components,
                                     n_features=n_features,
                                     n_nonzero_coefs=n_nonzero_coefs,
                                     random_state=0)

    for c in true_code.T:
        assert_true(np.count_nonzero(c) == n_nonzero_coefs)

#------------------------------------------------------------
# Test for _projfunc
#------------------------------------------------------------

def test_constraint():
    nn = 1
    N = 1000
    ds = 0.1
    i_s = 0.8

    # Take a random vector and project it onto desired sparseness
    x = np.random.randn(N, 1)
    x = x / np.linalg.norm(x)
    k1 = np.sqrt(N) - (np.sqrt(N)-1) * ds
    x, usediters = _projfunc(x, k1, 1, nn=nn)

    # Take another random vector and project to initial sparseness
    s = np.random.randn(N, 1)
    s = s / np.linalg.norm(s)
    k1 = np.sqrt(N) - (np.sqrt(N)-1) * i_s
    s, usediters = _projfunc(s, k1, 1, nn=nn)

    # Project s to achieve desired sparseness, save 'usediters'
    k1 = np.sqrt(N) - (np.sqrt(N)-1) * ds
    v, usediters = _projfunc(s, k1, 1, nn=nn)

    print(v)
    # Expect satisfying L1 and L2 constraints
    assert_true(abs(np.sum(abs(v)) - k1) < 1e-8) and (abs(np.sum(v**2)-1) < 1e-8)
    # Expect satisfying nonnegativity
    if nn:
        assert_true(np.min(v) >= 0)

    # Expect satisfying closest point
    assert_true(np.linalg.norm(x - s) > (np.linalg.norm(v - s) - 1e-10))


def test_simple_value():
    nn = 1
    N = 1500
    s = np.random.randn(N, 1)
    k1 = np.sqrt(N) - (np.sqrt(N)-1) * 0.8
    x, usediters = _projfunc(s, k1, 1, nn=nn)
    assert_true(np.min(x) >= 0)


def test_sparsity():
    nn = 1
    N = 1500
    s = np.random.randn(N, 1)
    for i in range(1, 10):
        d_s = i * 0.1
        k1 = np.sqrt(N) - (np.sqrt(N)-1) * d_s
        x, usediters = _projfunc(s, k1, 1, nn=nn)
        sp = _hoyers_sparsity(x)
        assert_almost_equal(d_s, sp, decimal=2)


def test_k1_value():
    nn = 1
    N = 1500
    s = np.random.randn(N, 1)
    k1 = 0.9
    msg = "k1 should greater than 1.0; but k1 was given %r." % k1
    assert_raise_message(ValueError, msg, _projfunc, s, k1, 1, nn=nn)


#------------------------------------------------------------
# Test for Hoyer's sparsity
#------------------------------------------------------------
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

    assert_true(abs(ds - np.mean(sp) < 0.2))
