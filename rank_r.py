import numpy as np
from sklearn.utils.extmath import randomized_svd
import time

def svds(A, m):
    U, Sigma, VT = randomized_svd(A, n_components = m, random_state = None)
    return U, Sigma, VT

def cal_sketch(tensor, k, side):
    if side == 0:
        A = tensor.ten2mat(tensor.X, 1)
        U, __, __ = tensor.svds(A, k)
        S = U.T
        return S
    elif side == 1:
        A = tensor.ten2mat(tensor.X, 2)
        U, __, __ = tensor.svds(A, k)
        W = U
        return W
    elif side == 2:
        U = tensor.HOOI(k)
        return [U[0].T, U[1].T]
    else:
        print("Value of side must be  0 or 1 or 2!")
        raise ValueError 

def Gaussian_sketch(tensor, k, side):
    m, n = tensor.shape[1], tensor.shape[2]
    if side == 0:
        return (np.random.randn(k, m))
    elif side == 1:
        return (np.random.randn(n, k))
    elif side == 2:
        ans = []
        ans.append(np.random.randn(m, k))
        ans.append(np.random.randn(n, k))
        return ans 
    else:
        print("Value of side must be 0 or 1 or 2!")
        raise ValueError

def Sparse_sketch(tensor, k, side):
    m, n = tensor.shape[1], tensor.shape[2]
    S = np.zeros((k, m))
    W = np.zeros((n, k))

    if side == 0:
        h = np.random.randint(0, k, size = m)
        for i in range(m):
            S[h[i], i] = 1
        return S
    elif side == 1:
        h = np.random.randint(0, k, size = n)
        for i in range(n):
            W[h[i], i] = 1
        return W
    elif side == 2:
        h1 = np.random.randint(0, k, size = m)
        for i in range(m):
            S[h1[i], i] = 1
        h2 = np.random.randint(0, k, size = n)
        for i in range(n):
            W[h2[i], i] = 1
        return [S, W]
    else:
        print("Value of side must be 0 or 1 or 2!")
        raise ValueError    

def cal_lowrank(U, A, r, side):
    if side == 0:
        S = U
        Q, __ = np.linalg.qr((A.T)@(S.T))
        U1, Sigma, VT = svds(A@Q, r)
        Sigma = np.diag(Sigma)
        return U1@Sigma@VT@(Q.T)
    elif side == 1:
        W = U
        Q, __ = np.linalg.qr(A@W)
        U1, Sigma, VT = svds((Q.T)@A, r)
        Sigma = np.diag(Sigma)
        return Q@U1@Sigma@VT
    elif side == 2:
        Q = U[0].T
        P = U[1].T
        Q, __ = np.linalg.qr(A.T@Q)
        P, __ = np.linalg.qr(A@P)
        U1, Sigma, VT = svds(P.T@A@Q, r)
        return P@U1@np.diag(Sigma)@VT@(Q.T)       
    else:
        print("Value of side must be  0 or 1 or 2!")
        raise ValueError

def error(Abest, Te, args, U):
    te_err = 0.0 # Average test error 
    timeR_te = 0.0 # Total time for randomized algorithms(SCW) : test data 
    for j in range(Te.shape[0]):
        A = Te[j]
        tic = time.time()
        A_approx = cal_lowrank(U, A, args.r, args.side)
        toc = time.time()
        timeR_te += toc - tic 
        te_err += (np.linalg.norm(A - A_approx) - np.linalg.norm(A - Abest[j]))/np.linalg.norm(A - Abest[j])
    return [te_err/Te.shape[0], timeR_te]

