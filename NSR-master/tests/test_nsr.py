import numpy as np
import scipy.sparse as sp
import numbers

import warnings
from scipy import linalg
from nsr import NSR, nonnegative_sparse_representation
import nsr # For testing internals
from sklearn.decomposition import nmf
from scipy.sparse import csc_matrix
from sklearn.datasets import make_sparse_coded_signal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_raise_message, assert_no_warnings
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.extmath import squared_norm
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning


def test_count_atoms():

    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))

    rng = np.random.mtrand.RandomState(43)
    B = np.abs(rng.randn(10, 10))

    C = A.copy()
    C[:, :5] = np.zeros((10, 5))

    recovered_0 = nsr._count_atoms(A, B)
    assert_true(recovered_0 == 0.0)

    recovered_100 = nsr._count_atoms(A, A)
    assert_true(recovered_100 == 100.0)

    recovered_50 = nsr._count_atoms(A, C)
    assert_true(recovered_50 == 50.0)

def test_initialize_nn_output():
    # Test that initialization does not return negative values
    rng = np.random.mtrand.RandomState(42)
    data = np.abs(rng.randn(10, 10))
    for init in ('random', 'nndsvd', 'nndsvda', 'nndsvdar'):
        W, H = nmf._initialize_nmf(data, 10, init=init, random_state=0)
        assert_false((W < 0).any() or (H < 0).any())


def test_parameter_checking():
    A = np.ones((2, 2))
    name = 'spam'
    msg = "Invalid solver parameter: got 'spam' instead of one of"
    assert_raise_message(ValueError, msg, NSR(solver=name).fit, A)
    msg = "Invalid init parameter: got 'spam' instead of one of"
    assert_raise_message(ValueError, msg, NSR(init=name).fit, A)
    msg = "Invalid beta_loss parameter: got 'spam' instead of one"
    assert_raise_message(ValueError, msg, NSR(solver='mu',
                                              beta_loss=name).fit, A)
#    msg = "Invalid beta_loss parameter: solver 'cd' does not handle "
#    msg += "beta_loss = 1.0"
#    assert_raise_message(ValueError, msg, NMF(solver='cd',
#                                              beta_loss=1.0).fit, A)
    msg = "Negative values in data passed to"
    assert_raise_message(ValueError, msg, NSR().fit, -A)
    assert_raise_message(ValueError, msg, nmf._initialize_nmf, -A,
                         2, 'nndsvd')
    clf = NSR(2, tol=0.1).fit(A)
    assert_raise_message(ValueError, msg, clf.transform, -A)

def test_initialize_close():
    # Test NNDSVD error
    # Test that _initialize_nmf error is less than the standard deviation of
    # the entries in the matrix.
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    W, H = nmf._initialize_nmf(A, 10, init='nndsvd')
    error = linalg.norm(np.dot(W, H) - A)
    sdev = linalg.norm(A - A.mean())
    assert_true(error <= sdev)


def test_initialize_variants():
    # Test NNDSVD variants correctness
    # Test that the variants 'nndsvda' and 'nndsvdar' differ from basic
    # 'nndsvd' only where the basic version has zeros.
    rng = np.random.mtrand.RandomState(42)
    data = np.abs(rng.randn(10, 10))
    W0, H0 = nmf._initialize_nmf(data, 10, init='nndsvd')
    Wa, Ha = nmf._initialize_nmf(data, 10, init='nndsvda')
    War, Har = nmf._initialize_nmf(data, 10, init='nndsvdar',
                                   random_state=0)

    for ref, evl in ((W0, Wa), (W0, War), (H0, Ha), (H0, Har)):
        assert_almost_equal(evl[ref != 0], ref[ref != 0])


# ignore UserWarning raised when both solver='mu' and init='nndsvd'
@ignore_warnings(category=UserWarning)
def test_nsr_fit_nn_output():
    # Test that the decomposition does not contain negative values
    A = np.c_[5 * np.ones(5) - np.arange(1, 6),
              5 * np.ones(5) + np.arange(1, 6)]
#    for solver in ('cd', 'mu'):
    for solver in ['mu']:
        for init in (None, 'random'):
            model = NSR(n_components=2, solver=solver, init=init,
                        random_state=0)
            transf = model.fit_transform(A)
            assert_false((model.components_ < 0).any() or
                         (transf < 0).any())


def test_nsr_fit_close():
    rng = np.random.mtrand.RandomState(42)
    # Test that the fit is not too far away
#    for solver in ('cd', 'mu'):
    for solver in ['mu']:
        pnmf = NSR(5, solver=solver, init='random', random_state=0,
                   max_iter=600)
        X = np.abs(rng.randn(6, 5))
        assert_less(pnmf.fit(X).reconstruction_err_, 0.1)


def test_nsr_transform():
    # Test that NMF.transform returns close values
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(6, 5))
#    for solver in ['cd', 'mu']:
    for solver in ['mu']:
        m = NSR(solver=solver, n_components=3, init='random',
                random_state=0, tol=1e-5)
        ft = m.fit_transform(A)
        t = m.transform(A)
        assert_array_almost_equal(ft, t, decimal=2)


def test_nsr_transform_custom_init():
    # Smoke test that checks if NMF.transform works with custom initialization
    random_state = np.random.RandomState(0)
    A = np.abs(random_state.randn(6, 5))
    n_components = 4
    avg = np.sqrt(A.mean() / n_components)
    H_init = np.abs(avg * random_state.randn(n_components, 5))
    W_init = np.abs(avg * random_state.randn(6, n_components))

    m = NSR(solver='mu', n_components=n_components, init='custom',
            random_state=0)
    m.fit_transform(A, code=W_init, dictionary=H_init)
    m.transform(A)


def test_nsr_inverse_transform():
    # Test that NMF.inverse_transform returns close values
    random_state = np.random.RandomState(0)
    A = np.abs(random_state.randn(6, 4))
#    for solver in ('cd', 'mu'):
    for solver in ['mu']:
        m = NSR(solver=solver, n_components=4, init='random', random_state=0,
                max_iter=1000)
        ft = m.fit_transform(A)
        A_new = m.inverse_transform(ft)
        assert_array_almost_equal(A, A_new, decimal=2)


def test_nsr_fit_transform_logged():
    random_state = np.random.RandomState(0)
    A = np.abs(random_state.randn(6, 4))
    for solver in ['mu']:
        m = NSR(solver=solver, n_components=4, init='random', random_state=0,
                max_iter=1000)
        ft, logs = m.fit_transform_logged(A)
        assert_false((m.components_ < 0).any() or (ft < 0).any())
        assert_true(len(logs), 4)


def test_n_components_greater_n_features():
    # Smoke test for the case of more components than features.
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(30, 10))
    NSR(n_components=15, random_state=0, tol=1e-2).fit(A)


def test_nsr_sparse_input():
    # Test that sparse matrices are accepted as input
    from scipy.sparse import csc_matrix

    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    A[:, 2 * np.arange(5)] = 0
    A_sparse = csc_matrix(A)

#    for solver in ('cd', 'mu'):
    for solver in ['mu']:
        est1 = NSR(solver=solver, n_components=5, init='random',
                   random_state=0, tol=1e-2)
        est2 = clone(est1)

    W1 = est1.fit_transform(A)
    W2 = est2.fit_transform(A_sparse)
    H1 = est1.components_
    H2 = est2.components_

    assert_array_almost_equal(W1, W2)
    assert_array_almost_equal(H1, H2)


def test_nsr_sparse_transform():
    # Test that transform works on sparse data.  Issue #2124
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(3, 2))
    A[1, 1] = 0
    A = csc_matrix(A)
#    for solver in ('cd', 'mu'):
    for solver in ['mu']:
        model = NSR(solver=solver, random_state=0, n_components=2,
                    max_iter=400)
        A_fit_tr = model.fit_transform(A)
        A_tr = model.transform(A)
        assert_array_almost_equal(A_fit_tr, A_tr, decimal=1)


def test_nonnegative_sparse_representation_consistency():
    # Test that the function is called in the same way, either directly
    # or through the NMF class
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    A[:, 2 * np.arange(5)] = 0
#    for solver in ('cd', 'mu'):
    for solver in ['mu']:
        W_nmf, H, _ = nonnegative_sparse_representation(
            A, solver=solver, random_state=1, tol=1e-2)
        W_nmf_2, _, _ = nonnegative_sparse_representation(
            A, dictionary=H, update_dict=False, solver=solver, random_state=1, tol=1e-2)

        model_class = NSR(solver=solver, random_state=1, tol=1e-2)
        W_cls = model_class.fit_transform(A)
        W_cls_2 = model_class.transform(A)
        assert_array_almost_equal(W_nmf, W_cls, decimal=10)
        assert_array_almost_equal(W_nmf_2, W_cls_2, decimal=10)


def test_nonnegative_sparse_representation_checking():
    A = np.ones((2, 2))
    X = np.ones((5, 2))
    W = np.ones((5, 3))
    H = np.ones((3, 2))
    # Test parameters checking is public function
    nnmf = nonnegative_sparse_representation
    assert_no_warnings(nnmf, X, W, H)
    msg = ("Number of components must be a positive integer; "
           "got (n_components=1.5)")
    assert_raise_message(ValueError, msg, nnmf, A, A, A, 1.5)
    msg = ("Number of components must be a positive integer; "
           "got (n_components='2')")
    assert_raise_message(ValueError, msg, nnmf, A, A, A, '2')
    msg = "Negative values in data passed to NMF (input H)"
    assert_raise_message(ValueError, msg, nnmf, A, A, -A, 2, 'custom')
    msg = "Negative values in data passed to NMF (input W)"
    assert_raise_message(ValueError, msg, nnmf, A, -A, A, 2, 'custom')
    msg = "Array passed to NMF (input H) is full of zeros"
    assert_raise_message(ValueError, msg, nnmf, A, A, 0 * A, 2, 'custom')
    msg = "Invalid regularization parameter: got 'spam' instead of one of"
    assert_raise_message(ValueError, msg, nnmf, A, A, 0 * A, 2, 'custom', True,
                         'mu', 2., 1e-4, 200, 0., 0., 'spam')


def _beta_divergence_dense(X, W, H, beta):
    """Compute the beta-divergence of X and W.H for dense array only.

    Used as a reference for testing nmf._beta_divergence.
    """
    if isinstance(X, numbers.Number):
        W = np.array([[W]])
        H = np.array([[H]])
        X = np.array([[X]])

    WH = np.dot(W, H)

    if beta == 2:
        return squared_norm(X - WH) / 2

    WH_Xnonzero = WH[X != 0]
    X_nonzero = X[X != 0]
    np.maximum(WH_Xnonzero, 1e-9, out=WH_Xnonzero)

    if beta == 1:
        res = np.sum(X_nonzero * np.log(X_nonzero / WH_Xnonzero))
        res += WH.sum() - X.sum()

    elif beta == 0:
        div = X_nonzero / WH_Xnonzero
        res = np.sum(div) - X.size - np.sum(np.log(div))
    else:
        res = (X_nonzero ** beta).sum()
        res += (beta - 1) * (WH ** beta).sum()
        res -= beta * (X_nonzero * (WH_Xnonzero ** (beta - 1))).sum()
        res /= beta * (beta - 1)

    return res


def test_hoyers_sparsity():

    n_components, n_features = 512, 100
    n_samples = 100
    n_nonzero_coefs = 17
    _, _, sparse_mat = make_sparse_coded_signal(n_samples=n_samples,
                                                n_components=n_components,
                                                n_features=n_features,
                                                n_nonzero_coefs=n_nonzero_coefs,
                                                random_state=0)

    rng = np.random.mtrand.RandomState(42)
    non_sparse_mat = rng.randn(n_components, n_samples)

    high_sp = nsr._hoyers_sparsity(sparse_mat)
    low_sp = nsr._hoyers_sparsity(non_sparse_mat)

    assert_greater(np.mean(high_sp), 0.9)
    assert_less(np.mean(low_sp), 0.3)


def test_beta_divergence():
    # Compare _beta_divergence with the reference _beta_divergence_dense
    n_samples = 20
    n_features = 10
    n_components = 5
    beta_losses = [0., 0.5, 1., 1.5, 2.]

    # initialization
    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    X[X < 0] = 0.
    X_csr = sp.csr_matrix(X)
    W, H = nmf._initialize_nmf(X, n_components, init='random', random_state=42)

    for beta in beta_losses:
        ref = _beta_divergence_dense(X, W, H, beta)
        loss = nmf._beta_divergence(X, W, H, beta)
        loss_csr = nmf._beta_divergence(X_csr, W, H, beta)

        assert_almost_equal(ref, loss, decimal=7)
        assert_almost_equal(ref, loss_csr, decimal=7)


def test_special_sparse_dot():
    # Test the function that computes np.dot(W, H), only where X is non zero.
    n_samples = 10
    n_features = 5
    n_components = 3
    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    X[X < 0] = 0.
    X_csr = sp.csr_matrix(X)

    W = np.abs(rng.randn(n_samples, n_components))
    H = np.abs(rng.randn(n_components, n_features))

    WH_safe = nmf._special_sparse_dot(W, H, X_csr)
    WH = nmf._special_sparse_dot(W, H, X)

    # test that both results have same values, in X_csr nonzero elements
    ii, jj = X_csr.nonzero()
    WH_safe_data = np.asarray(WH_safe[ii, jj]).ravel()
    assert_array_almost_equal(WH_safe_data, WH[ii, jj], decimal=10)

    # test that WH_safe and X_csr have the same sparse structure
    assert_array_equal(WH_safe.indices, X_csr.indices)
    assert_array_equal(WH_safe.indptr, X_csr.indptr)
    assert_array_equal(WH_safe.shape, X_csr.shape)


@ignore_warnings(category=ConvergenceWarning)
def test_nsr_multiplicative_update_sparse():
    # Compare sparse and dense input in multiplicative update NMF
    # Also test continuity of the results with respect to beta_loss parameter
    n_samples = 20
    n_features = 10
    n_components = 5
    alpha = 0.1
    l1_ratio = 0.5
    n_iter = 20

    # initialization
    rng = np.random.mtrand.RandomState(1337)
    X = rng.randn(n_samples, n_features)
    X = np.abs(X)
    X_csr = sp.csr_matrix(X)
    W0, H0 = nmf._initialize_nmf(X, n_components, init='random',
                                 random_state=42)

    for beta_loss in (-1.2, 0, 0.2, 1., 2., 2.5):
        # Reference with dense array X
        W, H = W0.copy(), H0.copy()
        W1, H1, _ = nonnegative_sparse_representation(
            X, W, H, n_components, init='custom', update_dict=True,
            solver='mu', beta_loss=beta_loss, max_iter=n_iter, alpha=alpha,
            l1_ratio=l1_ratio, regularization='both', random_state=42)

        # Compare with sparse X
        W, H = W0.copy(), H0.copy()
        W2, H2, _ = nonnegative_sparse_representation(
            X_csr, W, H, n_components, init='custom', update_dict=True,
            solver='mu', beta_loss=beta_loss, max_iter=n_iter, alpha=alpha,
            l1_ratio=l1_ratio, regularization='both', random_state=42)

        assert_array_almost_equal(W1, W2, decimal=7)
        assert_array_almost_equal(H1, H2, decimal=7)

        # Compare with almost same beta_loss, since some values have a specific
        # behavior, but the results should be continuous w.r.t beta_loss
        beta_loss -= 1.e-5
        W, H = W0.copy(), H0.copy()
        W3, H3, _ = nonnegative_sparse_representation(
            X_csr, W, H, n_components, init='custom', update_dict=True,
            solver='mu', beta_loss=beta_loss, max_iter=n_iter, alpha=alpha,
            l1_ratio=l1_ratio, regularization='both', random_state=42)

        assert_array_almost_equal(W1, W3, decimal=4)
        assert_array_almost_equal(H1, H3, decimal=4)


def test_nsr_negative_beta_loss():
    # Test that an error is raised if beta_loss < 0 and X contains zeros.
    # Test that the output has not NaN values when the input contains zeros.
    n_samples = 6
    n_features = 5
    n_components = 3

    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    X[X < 0] = 0
    X_csr = sp.csr_matrix(X)

    def _assert_nmf_no_nan(X, beta_loss):
        W, H, _ = nonnegative_sparse_representation(
            X, n_components=n_components, solver='mu', beta_loss=beta_loss,
            random_state=0, max_iter=1000)
        assert_false(np.any(np.isnan(W)))
        assert_false(np.any(np.isnan(H)))

    msg = "When beta_loss <= 0 and X contains zeros, the solver may diverge."
    for beta_loss in (-0.6, 0.):
        assert_raise_message(ValueError, msg, _assert_nmf_no_nan, X, beta_loss)
        _assert_nmf_no_nan(X + 1e-9, beta_loss)

    for beta_loss in (0.2, 1., 1.2, 2., 2.5):
        _assert_nmf_no_nan(X, beta_loss)
        _assert_nmf_no_nan(X_csr, beta_loss)

""" ISSUE (Can not test regularization;
            because this test assumes 'nnsvd' initialize
            but 'random' initialize in overcomplete situation)
def test_nsr_regularization():
    # Test the effect of L1 and L2 regularizations
    n_samples = 6
    n_features = 5
    n_components = 3
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(n_samples, n_features))

    # L1 regularization should increase the number of zeros
    l1_ratio = 1.
#    for solver in ['cd', 'mu']:
    for solver in ['mu']:
        regul = nsr.NSR(n_components=n_components, solver=solver,
                        alpha=0.5, l1_ratio=l1_ratio, random_state=42)
        model = nsr.NSR(n_components=n_components, solver=solver,
                        alpha=0., l1_ratio=l1_ratio, random_state=42)

        W_regul = regul.fit_transform(X)
        W_model = model.fit_transform(X)

        H_regul = regul.components_
        H_model = model.components_

        W_regul_n_zeros = W_regul[W_regul == 0].size
        W_model_n_zeros = W_model[W_model == 0].size
        H_regul_n_zeros = H_regul[H_regul == 0].size
        H_model_n_zeros = H_model[H_model == 0].size

        assert_greater(W_regul_n_zeros, W_model_n_zeros)
        assert_greater(H_regul_n_zeros, H_model_n_zeros)

    # L2 regularization should decrease the mean of the coefficients
    l1_ratio = 0.
#    for solver in ['cd', 'mu']:
    for solver in ['mu']:
        regul = nsr.NSR(n_components=n_components, solver=solver,
                        alpha=0.5, l1_ratio=l1_ratio, random_state=42)
        model = nsr.NSR(n_components=n_components, solver=solver,
                        alpha=0., l1_ratio=l1_ratio, random_state=42)

        W_regul = regul.fit_transform(X)
        W_model = model.fit_transform(X)

        H_regul = regul.components_
        H_model = model.components_

        assert_greater(W_model.mean(), W_regul.mean())
        assert_greater(H_model.mean(), H_regul.mean())
"""

@ignore_warnings(category=ConvergenceWarning)
def test_nsr_decreasing():
    # test that the objective function is decreasing at each iteration
    n_samples = 20
    n_features = 15
    n_components = 10
    alpha = 0.1
    l1_ratio = 0.5
    tol = 0.

    # initialization
    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    np.abs(X, X)
    W0, H0 = nmf._initialize_nmf(X, n_components, init='random',
                                 random_state=42)

    for beta_loss in (-1.2, 0, 0.2, 1., 2., 2.5):

        for solver in ['mu']:
        # for solver in ('cd', 'mu'):
            if solver != 'mu' and beta_loss != 2:
                # not implemented
                continue
            W, H = W0.copy(), H0.copy()
            previous_loss = None
            for _ in range(30):
                # one more iteration starting from the previous results
                W, H, _ = nonnegative_sparse_representation(
                    X, W, H, beta_loss=beta_loss, init='custom',
                    n_components=n_components, max_iter=1, alpha=alpha,
                    solver=solver, tol=tol, l1_ratio=l1_ratio, verbose=0,
                    regularization='both', random_state=0, update_dict=True)

                loss = nmf._beta_divergence(X, W, H, beta_loss)
                if previous_loss is not None:
                    assert_greater(previous_loss, loss)
                previous_loss = loss

def test_evaluation_logging():
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    model = NSR(max_iter=10, n_components=10)
    transf, logs = model.fit_transform_logged(A, true_dict=A)

    print('len:', len(logs['error']))

    assert_true(len(logs['time']) == 11)
    assert_true(len(logs['error']) == 11)
    assert_true(len(logs['sparsity']) == 11)
    assert_true(len(logs['atoms']) == 11)

