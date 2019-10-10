import numpy as np


def dif1_matrix(x):
    '''
    In case for len(x) = 5:
    dif = np.array(
    [[1, 0, 0, 0, 0],
     [-1, 1, 0, 0, 0],
     [0, -1, 1, 0, 0],
     [0, 0, -1, 1, 0],
     [0, 0, 0, -1, 1]])
    '''
    # Diagonal elements are 1.
    dif_now = np.diag(np.ones(x))

    # Left elements of diagonal are -1.
    dif_pre_ones = np.ones(x - 1) * - 1  # -1 vector.
    dif_pre = np.diag(dif_pre_ones, k=-1)  # Diagonal matrix shiftedto left.
    dif = dif_now + dif_pre
    return dif

if __name__ == '__main__':
    mat = dif1_matrix(5)