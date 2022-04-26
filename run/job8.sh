#!/bin/bash -l
#SBATCH -o job%j.out
#SBATCH -t 72:00:00   
#SBATCH --partition=C072M0512G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
source activate torch
cd ..
python main.py --sketchType Tensorb --dataType cam --N 100 --N_train 80 --N_test 20 --raw 1 --bestdone 0
python main.py --sketchType Tensorb --dataType cam --N 100 --N_train 80 --N_test 20 --side 2
python main.py --sketchType Dlearning --dataType cam --N 100 --N_train 80 --N_test 20 
python main.py --sketchType Dlearning2 --dataType cam --N 100 --N_train 80 --N_test 20 --side 2
 
