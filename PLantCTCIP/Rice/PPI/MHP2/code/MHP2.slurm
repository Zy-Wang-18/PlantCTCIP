#!/bin/bash
#SBATCH -J MHP2
#SBATCH -p RTX3090-7
#SBATCH -N 1
#SBATCH -o MHP2.out
#SBATCH -e MHP2.err
python train_MH_RNAP2.py
