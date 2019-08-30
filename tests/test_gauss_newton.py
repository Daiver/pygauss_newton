import unittest

import numpy as np
import pygauss_newton


def oned_problem01_residuals(x):
    return x


def oned_problem01_jacobian(x):
    return np.array([[1]], dtype=np.float32)


class TestGaussNewton(unittest.TestCase):
    def test_1d_opt01(self):

        x = np.array([0.0])
        settings = pygauss_newton.Settings(verbose=False)
        res, res_state = pygauss_newton.gauss_newton(
            residuals_func=oned_problem01_residuals,
            jacobian_func=oned_problem01_jacobian,
            x0=x,
            settings=settings,
            update_functor=None
        )
        self.assertEqual(0, res_state.iter_ind)
        self.assertEqual(res_state.stopping_reason, pygauss_newton.StoppingReason.ByLossValue)
        self.assertEqual(0, res[0])

        x = np.array([2.0])
        settings = pygauss_newton.Settings(verbose=False)
        res, res_state = pygauss_newton.gauss_newton(
            residuals_func=oned_problem01_residuals,
            jacobian_func=oned_problem01_jacobian,
            x0=x,
            settings=settings,
            update_functor=None
        )
        self.assertEqual(1, res_state.iter_ind)
        self.assertEqual(res_state.stopping_reason, pygauss_newton.StoppingReason.ByLossValue)
        self.assertEqual(0, res[0])

        x = np.array([-20.0])
        settings = pygauss_newton.Settings(verbose=False)
        res, res_state = pygauss_newton.gauss_newton(
            residuals_func=oned_problem01_residuals,
            jacobian_func=oned_problem01_jacobian,
            x0=x,
            settings=settings,
            update_functor=None
        )
        self.assertEqual(1, res_state.iter_ind)
        self.assertEqual(res_state.stopping_reason, pygauss_newton.StoppingReason.ByLossValue)
        self.assertEqual(0, res[0])

    def test_stop_by_callback01(self):
        x = np.array([-20.0])
        settings = pygauss_newton.Settings(verbose=False)
        res, res_state = pygauss_newton.gauss_newton(
            residuals_func=oned_problem01_residuals,
            jacobian_func=oned_problem01_jacobian,
            x0=x,
            settings=settings,
            update_functor=lambda x, _: False
        )
        self.assertEqual(0, res_state.iter_ind)
        self.assertEqual(res_state.stopping_reason, pygauss_newton.StoppingReason.ByCallback)


if __name__ == '__main__':
    unittest.main()
