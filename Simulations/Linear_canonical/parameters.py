"""
This file contains the parameters for the simulations with linear canonical model
* Linear State Space Models with Full Information
    # v = 0, -10, -20 dB
    # scaling model dim to 5x5, 10x10, 20x20, etc
    # scalable trajectory length T
    # random initial state
* Linear SS Models with Partial Information
    # observation model mismatch
    # evolution model mismatch
"""

import torch

m = 2 # state dimension = 2, 5, 10, etc.
#n=2 # for the duplicate version
n = 2 # observation dimension = 2, 5, 10, etc.

##################################
### Initial state and variance ###
##################################
m1_0 = torch.zeros(m, 1) # initial state mean

#########################################################
### state evolution matrix F and observation matrix H ###
#########################################################
# F in canonical form
#F = torch.eye(m)
#F[0] = torch.ones(1,m)
##### Use the following format when using the duplicate version to keep m=2, n=2
F = torch.tensor([[1, 0.3],
                  [0.00, 1]], dtype=torch.float32)
#F = torch.tensor([[0.98, 0.09],
#                  [0.00, 0.98]], dtype=torch.float32)

##### Use the following format when using the duplicate version to keep m=4, n=4
#F = torch.tensor([    [0.98, 0.09,0.00,0.00],
#                      [0.00, 0.98,0.00,0.00],
#                      [0.00, 0.00, 0.98, 0.09],
#                      [0.00, 0.00, 0.00, 0.98]], dtype=torch.float32)

##### Use the following format when using the duplicate version to keep m=32, n=32
# Base 2x2 block
#F2 = torch.tensor([[0.98, 0.09],
#                   [0.00, 0.98]], dtype=torch.float32)
# Repeat it 16 times on the diagonal → 32x32
#F = torch.kron(torch.eye(16, dtype=torch.float32), F2)

##### Use the following format when using the duplicate version to keep m=17, n=17
#F = 0.98 * torch.eye(17, dtype=torch.float32)
#F[:16, 16] = 0.09  # last column, rows 0..15
##### Use the following format when using the duplicate version to keep m=9, n=9
#F = 0.98 * torch.eye(9, dtype=torch.float32)
#F[:8, 8] = 0.09  # last column, rows 0..7


if m == 2:
    ##### Use the following format when using the duplicate version to keep m=2, n=2
    H = torch.tensor([[1, 0.00],
                          [0.00, 0.00]], dtype=torch.float32)
    ##### Use the following format when using the duplicate version to keep m=4, n=4
    #H = torch.tensor([[1, 0.00,0.00,0.00],
    #                       [0.00, 0.00,0.00,0.00],
    #                       [0, 0.00, 1.00, 0.00],
    #                       [0, 0.00, 0.00, 0.00]], dtype=torch.float32)
    #####
    #H = torch.tensor([[1, 0.00]], dtype=torch.float32)
    ##### Use the following format when using the duplicate version to keep m=32, n=32
    #H2 = torch.tensor([[1, 0.00],
    #               [0.00, 0.00]], dtype=torch.float32)
    # Repeat it 16 times on the diagonal → 32x32
    #H = torch.kron(torch.eye(16, dtype=torch.float32), H2)
    ##### Use the following format when using the duplicate version to keep m=9, n=9
    #H = torch.eye(9, dtype=torch.float32)  # 17x17 identity
    #H[8, 8] = 0.0  # zero the last diagonal element


else:
    # H in reverse canonical form
    H = torch.zeros(n,n)
    H[0] = torch.ones(1,n)
    for i in range(n):
        H[i,n-1-i] = 1



#######################
### Rotated F and H ###
#######################
F_rotated = torch.zeros_like(F)
H_rotated = torch.zeros_like(H)
if(m==2):
    alpha_degree = 0 # was 10. rotation angle in degree
    rotate_alpha = torch.tensor([alpha_degree/180*torch.pi])
    cos_alpha = torch.cos(rotate_alpha)
    sin_alpha = torch.sin(rotate_alpha)
    rotate_matrix = torch.tensor([[cos_alpha, -sin_alpha],
                                [sin_alpha, cos_alpha]])

    F_rotated = torch.mm(F,rotate_matrix)
    H_rotated =  torch.mm(H,rotate_matrix)
else:
    F_rotated = F
    H_rotated = H
###############################################
### process noise Q and observation noise R ###
###############################################
# Noise variance takes the form of a diagonal matrix
Q_structure = torch.eye(m)
R_structure = torch.eye(n)


print("State Evolution Matrix:",F_rotated)
print("Observation Matrix:",H_rotated)