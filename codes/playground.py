"""PLACE TO PLAY AROUND HERE"""

import numpy as np
from scipy.sparse import spdiags
print(np.vstack((1,-2,1)))
print(spdiags((1,2,3), 1, 4, 4).todense())
print(spdiags([(1,2,3), (4,5,6), (7,8,9)], (1, 0, -1), 4, 4).todense())
# print(spdiags(np.array([[1, 2, 3, 4, 5]]*4), [3,2,1, 0], 5, 5).todense())

