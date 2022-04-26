from data import gen_hsi, gen_video, gen_cam, gen_brain, gen_cavity
import time
import argparse
import tensorly as tl
import ast
from tensor import Tensor
import rank_r, adapt_r
import sparse_learning, dense_learning, dense_learn2
import sys
import numpy as np
import pickle
import os

tl.set_backend("numpy")
rawdir = 'data/'

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process necessary parameters.')
    def add(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    add("--problemType", type = str, default = "rankr", help = "rankr|adaptr")
    add("--dataType", type = str, default = "hsi", help = "hsi|cam|logo|brain|cavity")
    add("--sketchType", type = str, default = "Tensorb", help = "Tensorb|Srandom|Drandom|Slearning|Dlearning|Dlearning2")
    add("--side", type = int, default = 0, choices = [0, 1, 2], help = "0->SA|1->AW|2->SA(W^T)")
    add("--N", type = int, default = 200, help = "range of train")
    add("--N_train", type = int, default = 100, help = "#train")
    add("--N_test", type = int, default = 100, help = "#test")
    add("--r", type = int, default = 10, help = "truncation for rank-r approximation")
    add("--k", type  = int, default = 20, help = "sketch size")
    add("--testin", type = ast.literal_eval, default = True, help = "True for testin, else False")
    add("--gap", default  = 0, help = "if testout, gap>0; else, gap<=0")
    add("--raw", type = int, default = 0, help = "1:True|0:False")
    add("--eps", type = float, default = 0.1, help = "error tolerance for eps-approximation")
    add("--q", type = float, default = 0.5, help = "eta/eps")
    add("--bestdone", type = int, default = 1, help = "1:True|0:False")
    add("--iter", type=int, default=3000, help="total iterations")
    add("--scale", type=int, default=500, help="scale")
    add("--lr_S", type=float, default=1, help="learning rate scale?")
    add("--stop", type=float, default=0.001, help="condition for termination")

    args = parser.parse_args()
    print(args)
    if args.dataType == "cam":
        A_train, A_test = gen_cam.data(rawdir, args.N, args.N_train, args.N_test, args.raw, args.dataType)
    elif args.dataType == "hsi":
        A_train, A_test = gen_hsi.data(rawdir, args.N, args.N_train, args.N_test, args.raw, args.dataType)
    elif args.dataType == "brain":
        A_train, A_test= gen_brain.data(rawdir, args.N, args.N_train, args.N_test, args.raw, args.dataType)
    elif args.dataType == "logo":
        A_train, A_test = gen_video.data(rawdir, args.N, args.N_train, args.N_test, args.raw, args.dataType)
    elif args.dataType == "friends":
        A_train, A_test = gen_video.data(rawdir,args.N,  args.N_train, args.N_test, args.raw, args.dataType)
    elif args.dataType == "wpc":
        A_train, A_test = gen_video.data(rawdir, args.N, args.N_train, args.N_test, args.raw, args.dataType)
    elif args.dataType == "cavity":
        A_train, A_test = gen_cavity.data(rawdir, args.N, args.N_train, args.N_test, args.raw, args.dataType)
    else:
        print("Please input correct datatype!")
        exit
    if not args.bestdone:
        print("Start calculating best approximation...")
        A_best = []
        timeE = 0.0
        for i in range(A_test.shape[0]):
            tic = time.time()
            u, s, vt = rank_r.svds(A_test[i], args.r)
            timeE += time.time() - tic
            A_best.append(u@np.diag(s)@vt)
        print("Best approximation calculating finished.")
        with open(rawdir + 'trainset/' + args.dataType + '_best' +'.dat', 'wb') as f:
                pickle.dump([A_best, timeE], f)
                print("Best approximation storing finished.")
    else:
        path = rawdir + 'trainset/' + args.dataType + '_best' +'.dat'
        with open(path, 'rb') as f:
            A_best, timeE = pickle.load(f)
            
    print("Train data size:", A_train.shape)
    print("Test data size:", A_test.shape)
    tr = Tensor(A_train)
    if args.problemType == "rankr":
        tic = time.time() 
        if args.sketchType == "Tensorb":
            U = rank_r.cal_sketch(tr, args.k, args.side)
        elif args.sketchType == "Drandom":
            U = rank_r.Gaussian_sketch(A_train, args.k, args.side)
        elif args.sketchType == "Srandom":
            U = rank_r.Sparse_sketch(A_train, args.k, args.side)
        elif args.sketchType == "Slearning":
            U = sparse_learning.train(A_train, args) # return sketch_vector, sketch_value
        elif args.sketchType == "Dlearning":
            U = dense_learning.train(A_train, args)
        elif args.sketchType == "Dlearning2" and args.side == 2:
            U = dense_learn2.train(A_train, args)
        else:
            print("Wrong sketch type!")
            exit
        toc = time.time()
        timeS = toc - tic
        # if args.sketchType == "Slearning" or args.sketchType == "Dlearning":
        #     fname = "_data="+str(args.dataType)+"_method="+str(args.sketchType)+"_k="+str(args.k)+"_iter="+str(args.iter)+".txt"
        #     with open(rawdir+"results/"+fname, 'wb') as f:
        #         np.savetxt(f, U)
        print("Time for calculating sketch matrix:", timeS)
        print("Time for exact SVD:", timeE)
        ans = rank_r.error(A_best, A_test, args, U)
        # with open(rawdir +'results/'+args.dataType+'_'+args.sketchType+'_'+str(args.side)+'_'+str(args.N_train)+ '.pkl', 'wb') as f:
        #     pickle.dump([ans, timeS, timeE, train_index], f)
        # with open(rawdir +'results/'+args.dataType+'_'+args.sketchType+'_'+str(args.side) + '.pkl', 'rb') as f:
        #     ans, timeS, timeE = pickle.load(f)
        with open(rawdir + 'results/result_sec2.txt','a') as f:
            f.write(str([args.sketchType] + [args.dataType] + [args.k] + [args.side] +[args.N_train] +[args.iter] +ans + [timeS] + [timeE]))
            f.write("\n")

    elif args.problemType == "adaptr":
        tic = time.time()
        if args.sketchType == "Tensorb":
            U, k = adapt_r.cal_sketch(tr, args.eps*args.q, args.side)
        elif args.sketchType == "Drandom":
            U = adapt_r.Gaussian_sketch(A_train, args.k, args.side)
        elif args.sketchType == "Srandom":
            A_train = A_train/args.scale
            U = adapt_r.Sparse_sketch(A_train, args.k, args.side)
            A_train = A_train*args.scale
        toc = time.time()
        timeS = toc - tic
        ans = adapt_r.error(A_train, A_test, args, U)
        print(ans)
        print("Time for calculating sketch matrix:", timeS)
    else:
        print("Wrong problemType!")
        sys.exit()



        




















