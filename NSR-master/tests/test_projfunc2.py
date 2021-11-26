import numpy as np
from nsr_utils import _projfunc as projfunc

nn = 1
N = 10
x = np.array([0.498117443123917,-0.250147745967313,0.120701165599073,-0.0733167842633556,0.363149925925690,-0.353954844241657,0.0105814433636021,0.179575821726580,0.357707364028597,0.501881554130832])
x = x[:, np.newaxis]
sp = 0.8
k1 = max(np.sqrt(N)-(np.sqrt(N)-1)*sp, 1)
print(sp, k1)

x, usediters = projfunc(x, k1, 1, nn, verbose=True)
print(x, usediters)