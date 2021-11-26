from nsr_utils import _projfunc
import numpy as np

x = np.array([[1.5],[1]])
print('x shape: ',  x.shape)
k1 = sum(abs(x))
k2 = sum(x ** 2)
result = _projfunc(x, k1, k2, nn=True)[0]
print('result:', result)