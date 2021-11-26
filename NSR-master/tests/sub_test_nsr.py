import numpy as np
import scipy.sparse as sp
import numbers

import warnings
from scipy import linalg
from nsr import NSR, nonnegative_sparse_representation
import nsr  # For testing internals
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
        regul = nmf.NMF(n_components=n_components, solver=solver,
                        alpha=0.5, l1_ratio=l1_ratio, random_state=42, init='random')
        model = nmf.NMF(n_components=n_components, solver=solver,
                        alpha=0., l1_ratio=l1_ratio, random_state=42, init='random')

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

    """
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
        """

    # L2 regularization should decrease the mean of the coefficients
    l1_ratio = 0.
#    for solver in ['cd', 'mu']:
    """
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