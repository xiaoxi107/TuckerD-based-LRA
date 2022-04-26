import os
from numpy.linalg import norm, qr
import numpy as np
import tensorly as tl
from sklearn.utils.extmath import randomized_svd

class Tensor:
    '''
    -----tensor basic-----
    -X: data of tensor(np.array)
    -shape: shape of tensor(1d np.array) 
    -ndim: number of dim of tensor(int)
    -ten2mat: matricization of tensor(2d np.array)
    -svds: truncated svd of a matrix(2d np.array)
    -----tensor decompositions-----
    rank r & adaptive r
    -t_HOSVD, st_HOSVD, HOOI
    -adaptive_tHOSVD, adaptive_HOOI
    '''
    def __init__(self, X):
        self.X = X
        self.shape = X.shape
        self.ndim = len(X.shape)
    def ten2mat(self, tensor, mode):
        return tl.unfold(tensor, mode)
    def svds(self, A, k):
        U, Sigma, VT = randomized_svd(A, n_components = k, random_state = None)
        return U, Sigma, VT
    def t_HOSVD(self, k):
        if self.ndim != 3:
            print("Please ensure X is a 3-d tensor!")
            raise ValueError
        Y = self.X.copy()
        ans = list()
        for i in range(1, self.ndim):
            A = tl.unfold(Y, i)
            U, __, __ = self.svds(A, k)
            ans.append(U)
        return ans
    def st_HOSVD(self, k):
        if self.ndim != 3:
            print("Please ensure X is a 3-d tensor!")
            raise ValueError
        Y = self.X.copy()
        trans_shape = list(self.shape)
        ans = list()
        for i in [1, 2]:
            Y = tl.unfold(Y, i)
            U, Sigma, VT = self.svds(Y, k)
            Y = np.diag(Sigma)@VT
            trans_shape[i] = k
            Y = tl.fold(Y, i, trans_shape)
            ans.append(U)
        return ans
    def HOOI(self, k):
        if self.ndim != 3:
            print("Please ensure X is a 3-d tensor!")
            raise ValueError
        Y = self.X
        U = []
        U.append(np.random.randn(Y.shape[1], k))
        U.append(np.random.randn(Y.shape[2], k))
        # U = self.st_HOSVD(k)
        iters = 1
        i = 0
        oshape = self.X.shape
        while i < iters:
            for j in [0, 1]:
                B = tl.unfold(self.X, j + 1)
                B = np.matmul(U[j].T, B)
                nshape = list(oshape)
                nshape[j+1] = U[j].shape[1]
                B = tl.fold(B, j+1,tuple(nshape))
                B = tl.unfold(B, 2-j)
                u, __, __ = self.svds(B, k)
                U[1 - j] = u
            i += 1
        return U
    
    def adaptive_tHOSVD(self, tol):
        if self.ndim != 3:
            print("Please ensure X is a 3d tensor!\n")
            raise ValueError
        Y = self.X.copy()
        ans = list()
        for i in range(1, self.ndim):
            A = self.ten2mat(Y, i)
            U, Sigma, __ = self.svds(A, A.shape[0])
            for j in range(len(Sigma)):
                if sum(Sigma[:j + 1]**2) >= (1 - tol**2)*norm(A, 'fro')**2:
                    break
            k = j + 1
            S = U[:, :k]
            ans.append(S)
        return ans
    def adaptive_HOOI(self, tol):
        if self.ndim != 3:
            print("Please ensure X is a 3d tensor!\n")
            raise ValueError
        m = 0
        Y = self.X.copy()
        # U = self.st_HOSVD(R0)
        U = []
        Q, __ = qr(np.random.randn(Y.shape[1], Y.shape[1]))
        U.append(Q)
        Q, __ = qr(np.random.randn(Y.shape[2], Y.shape[2]))
        U.append(Q) 
        G = np.tensordot(Y, U[0], [1, 0])
        G = np.tensordot(G, U[1], [1, 0])
        x = (1-tol**2)*tl.norm(Y)**2
        eta = tl.norm(G)**2 - x
        while eta > 0:
            for n in range(0, self.ndim - 1):
                B = np.tensordot(Y, U[1 - n], (2 - n, 0))
                Bi = self.ten2mat(B, 1)
                u, Sigma, vt = self.svds(Bi, Bi.shape[0])
                for i in range(len(Sigma)):
                    if sum(Sigma[:(i+1)]**2) >= x:
                        break
                m = i
                U[n] = u[:, :m]
                print(m)
            G = np.diag(Sigma[:m])@vt[:m, :]
            eta = tl.norm(G)**2 - x
        return U, m


        


