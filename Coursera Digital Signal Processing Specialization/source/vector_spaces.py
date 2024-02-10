import numpy as np
import pytest

v_0 = np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2]).T
v_1 = np.array([1 / 2, 1 / 2, -1 / 2, -1 / 2]).T
v_2 = np.array([1 / 2, -1 / 2, 1 / 2, -1 / 2]).T
v_3 = np.array([1 / 2, -1 / 2, -1 / 2, 1 / 2]).T

y = np.array([0.5, -0.5, -0.5, -1.5])


def test_inner_product():
    res = np.dot(v_0, v_1)
    assert 0. == pytest.approx(res)


def test_expansion_coefficients():
    M = np.c_[np.conjugate(v_0), np.conjugate(v_1), np.conjugate(v_2), np.conjugate(v_3)]
    # y = a * M
    # a = y / M
    a = M @ y
    np.testing.assert_almost_equal(a, [-1.,  1.,  1.,  0.])


def is_basis_of_R_4(matrix):
    eigenvalues, _eigenvecs = np.linalg.eig(matrix.T)
    # a zero eigenvalue indicates linear dependency
    for e in eigenvalues:
        if np.isclose(e, 0.0):
            return False
    return True


def test_is_basis_of_R_4():
    a = np.c_[y, v_0, v_2, v_3]
    assert is_basis_of_R_4(a)
    b = np.c_[y, v_1, v_2, v_3]
    assert is_basis_of_R_4(b)
    c = np.c_[y, v_0, v_1, v_2]
    assert not is_basis_of_R_4(c)
    d = np.c_[y, v_1, v_2, v_3 - 2 * v_1]
    assert is_basis_of_R_4(d)
