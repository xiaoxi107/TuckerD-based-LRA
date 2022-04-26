from numpy.linalg import norm, qr, svd
import numpy as np
from sklearn.utils.extmath import randomized_svd
import time
import random

def svds(A, m):
    U, Sigma, VT = randomized_svd(A, n_components = m, random_state = None)
    return U, Sigma, VT

def Gaussian_sketch(tensor, k, side):
    m, n = tensor.shape[1], tensor.shape[2]
    if side == 0:
        return (np.random.randn(k, m))
    elif side == 1:
        return (np.random.randn(k, n))
    elif side == 2:
        ans = []
        ans.append(np.random.randn(143, m))
        ans.append(np.random.randn(767, n))
        return ans 
    else:
        print("Value of side must be 0 or 1 or 2!")
        raise ValueError

def Sparse_sketch(tensor, k, side):
    m, n = tensor.shape[1], tensor.shape[2]
    values = [1.0, -1.0]
    S = np.zeros((k, m))
    W = np.zeros((k, n))
    if side == 0:
        h = np.random.randint(0, k, size = m)
        for i in range(m):
            S[h[i], i] = random.choice(values)
        return S
    elif side == 1:
        h = np.random.randint(0, k, size = n)
        for i in range(n):
            W[h[i], i] = random.choice(values)
        return W
    elif side == 2:
        h1 = np.random.randint(0, k, size = m)
        for i in range(m):
            S[h1[i], i] = random.choice(values)
        h2 = np.random.randint(0, k, size = n)
        for i in range(n):
            W[h2[i], i] = random.choice(values)
        return [S, W]
    else:
        print("Value of side must be 0 or 1 or 2!")
        raise ValueError    

# calculate sketch matrix
def cal_sketch(tensor, tol, side):
    if side == 0:
        A = tensor.ten2mat(tensor.X, 1)
        U, Sigma, __ = svds(A, A.shape[0])
        base = (1 - tol**2)*norm(A, 'fro')**2
        for i in range(len(Sigma)):
            if sum(Sigma[:i + 1]**2) >= base:
                break
        k = i + 1
        S = U[:, :k].T
        return S, k
    elif side == 1:
        A = tensor.ten2mat(tensor.X, 2)
        U, Sigma, __ = svds(A, A.shape[0])
        base = (1 - tol**2)*norm(A, 'fro')**2
        for i in range(len(Sigma)):
            if sum(Sigma[:i + 1]**2) >= base:
                break
        k = i + 1
        W = U[:, :k].T
        return W, k
    elif side == 2:
        U, k = tensor.adaptive_HOOI(tol)
        # U = tensor.adaptive_tHOSVD(tol)
        return [U[0].T, U[1].T], k
    else:
        print("Value of side must be  0 or 1 or 2!")
        raise ValueError

# SCW algorithm for fixed-eps error
def adaptive_SCW(U, A, eps, side):
    if side == 0:
        S = U
        Q, __ = qr((A.T)@(S.T))
        eps1 = np.sqrt(1-(1-eps**2)*norm(A, 'fro')**2/norm(A@Q, 'fro')**2)
        U1, Sigma, VT = svd(A@Q)
        for i in range(Sigma.shape[0]):
            if sum(Sigma[:i + 1]**2) >= (1 - eps1**2)*sum(Sigma**2):
                break
        k = i + 1
        return U1[:, :k]@np.diag(Sigma[:k])@VT[:k, :]@Q.T, k

    elif side == 1:
        W = U
        Q, __ = qr(A@W.T)
        eps1 = np.sqrt(1-(1-eps**2)*norm(A, 'fro')**2/norm(Q.T@A, 'fro')**2)
        U1, Sigma, VT = svd((Q.T)@A)
        for i in range(Sigma.shape[0]):
            if sum(Sigma[:i + 1]**2) >= (1 - eps1**2)*sum(Sigma**2):
                break
        k = i + 1
        return Q@U1[:, :k]@np.diag(Sigma[:k])@VT[:k, :], k

    elif side == 2:
        Q = U[0].T
        P = U[1].T
        Q, __ = qr(A.T@Q)
        P, __ = qr(A@P)
        eps1 = np.sqrt(1-(1-eps**2)*norm(A, 'fro')**2/norm(P.T@A@Q, 'fro')**2)
        U1, Sigma, VT = svd(P.T@A@Q)
        for i in range(Sigma.shape[0]):
            if sum(Sigma[:i + 1]**2) >= (1 - eps1**2)*sum(Sigma**2):
                break
        k = i + 1
        return P@(U1[:, :k]@np.diag(Sigma[:k])@VT[:k, :])@Q.T, k
    else:
        print("The value of side must be 0 or 1 or 2!")
        raise ValueError

def exact_r(A, eps):
    __, Sigma, __ = svd(A)
    for i in range(len(Sigma)):
        if sum(Sigma[:i + 1]**2) >= (1 - eps**2)*sum(Sigma**2):
            break
    return i + 1

def error(A_train, A_test, args, U):
    train_err = 0.0
    test_err = 0.0
    train_err_r = 0.0
    test_err_r = 0.0
    count_tr = 0
    count_te = 0
    timeR_tr = 0
    timeE_tr = 0
    timeR_te = 0
    timeE_te = 0
    
    for i in range(A_train.shape[0]):
        A = A_train[i]
        tic  = time.time()
        A_approx, r = adaptive_SCW(U, A, args.eps, args.side)
        toc = time.time()
        timeR_tr += toc - tic

        err = norm(A - A_approx)/norm(A)
        tic = time.time()
        r_op = exact_r(A, args.eps)
        toc = time.time()
        timeE_tr += toc - tic
        if err >= args.eps:
            count_tr += 1
        else:
            err_r = (r_op - r)/r_op
            train_err_r += err_r
        train_err += err

    for j in range(A_test.shape[0]):
        A = A_test[j]
        tic = time.time()
        A_approx, r = adaptive_SCW(U, A, args.eps, args.side)
        toc = time.time()
        timeR_te += toc - tic

        err = norm(A - A_approx)/norm(A)
        tic = time.time()
        r_op = exact_r(A, args.eps)
        toc = time.time()
        timeE_te += toc - tic

        if err >= args.eps:
            count_te += 1
        else:
            err_r = (r_op - r)/r_op
            test_err_r += err_r
        test_err += err
    return [count_tr/A_train.shape[0], train_err_r/(A_train.shape[0]-count_tr), timeR_tr, count_te/A_test.shape[0], test_err_r/(A_test.shape[0]-count_te), timeR_te]
    
