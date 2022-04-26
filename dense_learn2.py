import torch
import os
import pickle
import sys
from pathlib import Path
import time


def get_hostname():
    with open("/etc/hostname") as f:
        hostname=f.read()
    hostname=hostname.split('\n')[0]
    return hostname

def mysvd(init_A,k):
    if k>min(init_A.size(0),init_A.size(1)):
        k=min(init_A.size(0),init_A.size(1))
    d=init_A.size(1)
    x=[torch.Tensor(d).uniform_() for i in range(k)]
    for i in range(k):
        x[i].requires_grad=False
    def perStep(x,A):
        x2=A.t().mv(A.mv(x))
        x3=x2.div(torch.norm(x2))
        return x3
    U=[]
    S=[]
    V=[]
    Alist=[init_A]
    for kstep in range(k): #pick top k eigenvalues
        cur_list=[x[kstep]]   #current history
        for j in range(300):  #steps
            cur_list.append(perStep(cur_list[-1],Alist[-1]))  #works on cur_list
        V.append((cur_list[-1]/torch.norm(cur_list[-1])).view(1,cur_list[-1].size(0)))
        S.append((torch.norm(Alist[-1].mv(V[-1].view(-1)))).view(1))
        U.append((Alist[-1].mv(V[-1].view(-1))/S[-1]).view(1,Alist[-1].size(0)))
        Alist.append(Alist[-1]-torch.ger(Alist[-1].mv(cur_list[-1]), cur_list[-1]))
    return torch.cat(U,0).t(),torch.cat(S,0),torch.cat(V,0).t()


def train(A_train, args):
    print(args)
    r = args.r
    k = args.k
    l = k

    lr = 10
    lr = lr*args.lr_S
    m,n = A_train.shape[1], A_train.shape[2]
    N_train = len(A_train)
    print("Working on data ", args.dataType)
    print("Dim = ", m, n)
    print("N train = ", N_train)

    print_freq = 10
    S  = torch.randn(k, m).detach()
    S.requires_grad = True
    W = torch.randn(l, n).detach()
    W.requires_grad = True
    for epoch in range(args.iter+1):
        if (epoch + 1)%1000==0 and lr > 1:
            lr = lr * 0.3
        A = A_train[int(torch.randint(N_train, [1]).item())]/args.scale
        A = torch.from_numpy(A).float()
        SA = torch.mm(S, A)
        AW = torch.mm(A, W.T)
        __, __, V = mysvd(SA, SA.size(1))
        U, __, __ = mysvd(AW, AW.size(1))
        UtAV = (U.T)@A@V
        U1, Sigma1, V1 = mysvd(UtAV, k)
        ans = U@U1@(torch.diag(Sigma1))@(V1.T)@(V.T)
        loss = torch.norm(ans - A)**2
        loss.backward()
        if epoch % print_freq == 0:
            print(epoch, '.', loss.cpu().item(), end = ",")
            print("\n")

        S.data -= lr*S.grad.data
        W.data -= lr*W.grad.data
        S.grad.data.fill_(0)
        W.grad.data.fill_(0)

        del A, SA, AW, U, V, UtAV, U1, Sigma1, V1, ans, loss
    print("Generating sketch done!\n")
    return S.detach().cpu().numpy(), W.detach().cpu().numpy()
        





        

        


            


        
        

