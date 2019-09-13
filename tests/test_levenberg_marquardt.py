import unittest

import numpy as np
import pygauss_newton
from pygauss_newton import levenberg_marquardt

from problems import oned_problem01_jacobian, oned_problem01_residuals
from problems import convex_concave_problem01_residuals, convex_concave_problem01_jacobian


class TestLevenbergMarquardt(unittest.TestCase):
    def test_1d_opt01(self):
        x = np.array([0.0])
        settings = pygauss_newton.settings.Settings(verbose=False, loss_stop_threshold=1e-10)
        res, res_state = pygauss_newton.levenberg_marquardt(
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
        settings = pygauss_newton.settings.Settings(verbose=False)
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
        settings = pygauss_newton.settings.Settings(verbose=False)
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
        settings = pygauss_newton.settings.Settings(verbose=False)
        res, res_state = pygauss_newton.levenberg_marquardt(
            residuals_func=oned_problem01_residuals,
            jacobian_func=oned_problem01_jacobian,
            x0=x,
            settings=settings,
            update_functor=lambda x, _: False
        )
        self.assertEqual(0, res_state.iter_ind)
        self.assertEqual(res_state.stopping_reason, pygauss_newton.StoppingReason.ByCallback)

    def test_convex_concave01(self):
        x = np.array([-20.0])
        settings = pygauss_newton.settings.Settings(verbose=False)
        res, res_state = pygauss_newton.levenberg_marquardt(
            residuals_func=convex_concave_problem01_residuals,
            jacobian_func=convex_concave_problem01_jacobian,
            x0=x,
            settings=settings,
            update_functor=None
        )
        self.assertEqual(-1, res[0])


if __name__ == '__main__':
    unittest.main()
