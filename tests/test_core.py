import pytest
from src.encnumpy.core import enc_ndarray
import numpy as np


def test_add():
    x1 = np.array([[1, 2], [3, 4]])
    x2 = np.array([[5, 6], [7, 8]])
    enc_x1 = enc_ndarray(x1, x1.shape, str(x1.dtype))
    enc_x2 = enc_ndarray(x2, x2.shape, str(x2.dtype))
    result = enc_x1 + enc_x2
    expected = np.array([[6, 8], [10, 12]])
    assert np.array_equal(result.data, expected)


def test_subtract():
    x1 = np.array([[10, 20], [30, 40]])
    x2 = np.array([[1, 2], [3, 4]])
    enc_x1 = enc_ndarray(x1, x1.shape, str(x1.dtype))
    enc_x2 = enc_ndarray(x2, x2.shape, str(x2.dtype))
    result = enc_x1 - enc_x2
    expected = np.array([[9, 18], [27, 36]])
    assert np.array_equal(result.data, expected)

def test_multiply():
    x1 = np.array([[1, 2], [3, 4]])
    x2 = np.array([[5, 6], [7, 8]])
    enc_x1 = enc_ndarray(x1, x1.shape, str(x1.dtype))
    enc_x2 = enc_ndarray(x2, x2.shape, str(x2.dtype))
    result = enc_x1 * enc_x2
    expected = np.array([[5, 12], [21, 32]])
    assert np.array_equal(result.data, expected)

def test_divide():
    x1 = np.array([[10, 20], [30, 40]])
    x2 = np.array([[2, 4], [5, 10]])
    enc_x1 = enc_ndarray(x1, x1.shape, str(x1.dtype))
    enc_x2 = enc_ndarray(x2, x2.shape, str(x2.dtype))
    result = enc_x1 / enc_x2
    expected = np.array([[5.0, 5.0], [6.0, 4.0]])
    assert np.array_equal(result.data, expected)

def test_divide_by_zero():
    x1 = np.array([[10, 20], [30, 40]])
    x2 = np.array([[0, 1], [2, 0]])
    enc_x1 = enc_ndarray(x1, x1.shape, str(x1.dtype))
    enc_x2 = enc_ndarray(x2, x2.shape, str(x2.dtype))
    with pytest.raises(ZeroDivisionError):
        enc_x1 / enc_x2


def test_add_strings():
    x1 = np.array([["a", "b"], ["c", "d"]])
    x2 = np.array([["e", "f"], ["g", "h"]])
    enc_x1 = enc_ndarray(x1, x1.shape, str(x1.dtype))
    enc_x2 = enc_ndarray(x2, x2.shape, str(x2.dtype))
    result = enc_x1 + enc_x2
    expected = np.array([["ae", "bf"], ["cg", "dh"]])
    assert np.array_equal(result.data, expected)