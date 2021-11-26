# This code rewrite hoyer's nmf matlab code.
# Ref: http://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf

import numpy as np
import matplotlib.pyplot as plt
#from nsr_utils import _projfunc
from projfunc import projfunc as _projfunc
import pickle

# Enforce non-negativity?
nn = 1

# These are the various dimensionalities to test
dims = [2, 3, 5, 10, 50, 100, 500, 1000, 3000, 5000, 10000]

# These are the various desired sparseness levels
ds = [0.1, 0.3, 0.5, 0.7, 0.9]

# These are the various initial sparseness levels
i_s = [0.1, 0.3, 0.5, 0.7, 0.9]

# How many tests for each case?
ntests = 50
iters = np.zeros((len(ds), len(i_s), len(dims), ntests))

for dsiter in range(0, len(ds)):
    for isiter in range(0, len(i_s)):
        desiredsparseness = ds[dsiter]
        initialsparseness = i_s[isiter]
        print(desiredsparseness, initialsparseness)

        # Go through all the different dimensionalities
        for dimiter in range(0, len(dims)):

            N = dims[dimiter]
            for testcase in range(0, ntests):

                # Take a random vector and project it onto desired sparseness
                x = np.random.randn(N, 1)
                x = x / np.linalg.norm(x)
                k1 = np.sqrt(N) - (np.sqrt(N)-1) * desiredsparseness
                x, usediters = _projfunc(x, k1, 1, nn=True)

                # Take another random vector and project to initial sparseness
                s = np.random.randn(N, 1)
                s = s / np.linalg.norm(s)
                k1 = np.sqrt(N) - (np.sqrt(N)-1) * initialsparseness
                s, usediters = _projfunc(s, k1, 1, nn=True)

                # Project s to achieve desired sparseness, save 'usediters'
                k1 = np.sqrt(N) - (np.sqrt(N)-1) * desiredsparseness
                v, usediters = _projfunc(s, k1, 1, nn=True)
                iters[dsiter, isiter, dimiter, testcase] = usediters

                if (abs(np.sum(abs(v)) - k1) > 1e-8) | (abs(np.sum(v**2)-1) > 1e-8):
                    raise ValueError('L1 or L2 constraint not satisfied!!!!')
                if nn:
                    if np.min(v) < 0:
                        raise ValueError('Positivity constraint not satisfied!!! v: %s', v)

                if np.linalg.norm(x - s) < (np.linalg.norm(v - s) - 1e-10):
                    raise ValueError('Not closest point!!!! Fatal error!!!')


# Show average number of iterations as a function of desired sparseness
# and initial sparseness
meaniters = np.mean(iters, axis=3)
meaniters = np.mean(meaniters, axis=2)

# Note: along the diagonal we really should have zeros, since if the
# initial and the desired sparsenesses match then there is no need for
# even a single iteration! Instead, we have numbers slightly larger than
# one, probably because of roundoff errors.

# Clearly, the 'worst case' is when the desired sparseness is high (0.9)
# and the initial sparseness is very low (0.1). In the paper we plot the
# average number of iterations required for this worst case scenario:

worstcase = iters[4, 0, :, :].reshape(len(dims), ntests)
meanworstcase = np.mean(worstcase, axis=1)
maxworstcase = np.max(worstcase, axis=1)
minworstcase = np.min(worstcase, axis=1)


plt.semilogx(dims, meanworstcase, 'k-')
plt.semilogx(dims, maxworstcase, 'k:')
plt.semilogx(dims, minworstcase, 'k:')
plt.ylabel('iterations required')
plt.xlabel('dimensionality')
plt.title('Number of iterations required \n for the projection algorithm to convergence')


plt.savefig('result_test_projfunc.png')
print('The result was saved as result_test_projfunc.png')

