#!/bin/bash
#SBATCH -J MH34
#SBATCH -p RTX4090-1
#SBATCH -N 1
#SBATCH -o MH34.out
#SBATCH -e MH34.err
python train_MH_H3K4.py
