import numpy as np


def oned_problem01_residuals(x):
    return x


def oned_problem01_jacobian(x):
    return np.array([[1]], dtype=np.float32)


def convex_concave_problem01_residuals(x):
    return np.array([x[0]**2 - 1.0], dtype=np.float32)


def convex_concave_problem01_jacobian(x):
    return np.array([[2.0 * x[0]]], dtype=np.float32)
