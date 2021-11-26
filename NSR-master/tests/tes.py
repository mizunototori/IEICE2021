import numpy as np
from numpy.matlib import repmat
from nsr_utils import _projfunc
from nsr_utils import _hoyers_sparsity
from nsr import _make_nn_sparse_coded_signal
from sklearn.preprocessing import normalize
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_almost_equal


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
for c in true_code.T:
    print(c)
print(np.mean(sp))
