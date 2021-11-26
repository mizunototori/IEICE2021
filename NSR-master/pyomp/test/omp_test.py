import unittest
import numpy as np

import sys
sys.path.append('..')
from omp import *

class TestData(object):
    '''Generates and stores test data for unit tests.'''
    def __init__(self, n=64):
        self.t = np.linspace(0, 1, n)
        self.X = np.array([np.cos(2*np.pi*m*self.t) for m in range(4)]).T
        self.coefp = [0., 1., 0., .5]
        self.coefn = [0., 1., 0., -.5]
        self.coefc = [1., 0., 0., 0.]
        self.yp = np.dot(self.X, self.coefp)
        self.yn = np.dot(self.X, self.coefn)
        self.yc = np.ones(self.yp.shape, dtype=float)

        
def testOmp(*args, **kwargs):
    '''Wrapper for testing omp that sets different defaults.'''
    for key in ['verbose']:
        if key not in kwargs.keys():
            kwargs[key] = False
    return omp(*args, **kwargs)

        
class OMPTestCase(unittest.TestCase):
    '''Tests of OMP implementation.'''
    
    def test_omp_nostd_nonneg(self):
        '''Test omp when no standardization is used and non-negative solutions 
            sought.
        '''
        # first part, answer should match data.coefp
        data = TestData()
        result = testOmp(data.X, data.yp)
        self.assertEqual(all(result.active == np.array([1,3])), True)
        err = np.linalg.norm(result.coef - data.coefp)
        self.assertEqual(err < 1e-14, True)
        
        # second, negative coefficient in input data should be 0 in omp solution
        result = testOmp(data.X, data.yn)
        self.assertEqual(all(result.active == np.array([1])), True)
        
    def test_omp_nostd_neg(self):
        '''Test omp when no standardization is used an unconstrained solutions 
            are sought.
        '''
        data = TestData()
        result = testOmp(data.X, data.yn, nonneg=False)
        self.assertEqual(all(result.active == np.array([1, 3])), True)
        err = np.linalg.norm(result.coef - data.coefn)
        self.assertEqual(err < 1e-14, True)

    def test_omp_k1(self):
        '''Test that one coefficient is found if one is requested.
        '''
        data = TestData()
        result = testOmp(data.X, data.yp, ncoef=1)
        self.assertEqual(all(result.active == np.argmax(data.coefp)), True)

    def test_omp_tol(self):   
        '''Test that the algorithm stops after getting desired accuracy.'''
        data = TestData()
        tol = 0.5 / np.linalg.norm(data.yp) * np.sqrt(len(data.yp)) 
        result = testOmp(data.X, data.yp, tol=tol)
        self.assertEqual(len(result.active) == 1, True)    

    def test_omp_maxit(self):
        '''Test that iteration stops after maxit loops
        '''
        data = TestData()
        result = testOmp(data.X, data.yp, maxit=1)
        self.assertEqual(len(result.active) == 1, True)
        self.assertEqual(len(result.err) == 1, True)

        
if __name__ == '__main__':
    unittest.main()
