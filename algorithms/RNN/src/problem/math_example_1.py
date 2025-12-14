import numpy as np

def f(x):
    """ Hàm mục tiêu: f(x) """
    x1, x2 = x[0], x[1]
    tu = x1**2 + x2**2 + 3
    mau = 1 + 2*x1 + 8*x2
    return tu / mau

def grad_f(x):
    """ Đạo hàm f(x) """
    x1, x2 = x[0], x[1]
    tu = x1**2 + x2**2 + 3
    mau = 1 + 2*x1 + 8*x2

    # Đạo hàm u'v - uv' / v^2
    # u = x1^2 + x2^2 + 3  => u'x1 = 2x1, u'x2 = 2x2
    # v = 1 + 2x1 + 8x2    => v'x1 = 2,   v'x2 = 8

    df_dx1 = (2*x1 * mau - tu * 2) / (mau**2)
    df_dx2 = (2*x2 * mau - tu * 8) / (mau**2)
    return np.array([df_dx1, df_dx2])

def g(x):
    """
    Ràng buộc: -x1^2 - 2x1x2 <= -4
    Chuyển vế thành g(x) <= 0:  4 - x1^2 - 2x1x2 <= 0
    """
    x1, x2 = x[0], x[1]
    return 4 - x1**2 - 2*x1*x2

def grad_g(x):
    """ Đạo hàm g(x) """
    x1, x2 = x[0], x[1]
    # dg/dx1 = -2x1 - 2x2
    dg_dx1 = -2*x1 - 2*x2
    # dg/dx2 = -2x1
    dg_dx2 = -2*x1
    return np.array([dg_dx1, dg_dx2])