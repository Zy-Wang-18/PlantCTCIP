#!/bin/bash
#SBATCH -J ZS34
#SBATCH -p RTX4090-1
#SBATCH -N 1
#SBATCH -o ZS34.out
#SBATCH -e ZS34.err
python train_ZS_H3K4.py
