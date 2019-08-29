import unittest

import numpy as np
import pygauss_newton


class TestGaussNewton(unittest.TestCase):
    def test_1d_opt01(self):
        def residuals(x):
            return x

        def jacobian(x):
            return np.array([[1]], dtype=np.float32)

        x = np.array([0.0])
        settings = pygauss_newton.Settings(verbose=False)
        res = pygauss_newton.gauss_newton(
            residuals_func=residuals,
            jacobian_func=jacobian,
            x0=x,
            settings=settings,
            update_functor=None
        )
        self.assertEqual(0, res[0])

        x = np.array([2.0])
        settings = pygauss_newton.Settings(verbose=False)
        res = pygauss_newton.gauss_newton(
            residuals_func=residuals,
            jacobian_func=jacobian,
            x0=x,
            settings=settings,
            update_functor=None
        )
        self.assertEqual(0, res[0])

        x = np.array([-20.0])
        settings = pygauss_newton.Settings(verbose=False)
        res = pygauss_newton.gauss_newton(
            residuals_func=residuals,
            jacobian_func=jacobian,
            x0=x,
            settings=settings,
            update_functor=None
        )
        self.assertEqual(0, res[0])


if __name__ == '__main__':
    unittest.main()
