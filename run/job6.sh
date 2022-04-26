#!/bin/bash -l
#SBATCH -o job%j.out
#SBATCH -t 72:00:00   
#SBATCH --partition=C072M0512G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
source activate torch
cd ..
python main.py --sketchType Tensorb --dataType logo --bestdone 0 --N 200 --N_train 40 --raw 1
python main.py --sketchType Tensorb --dataType logo --bestdone 0 --N 200 --N_train 40 --side 2
python main.py --sketchType Dlearning --dataType logo --bestdone 0 --N 200 --N_train 40 


