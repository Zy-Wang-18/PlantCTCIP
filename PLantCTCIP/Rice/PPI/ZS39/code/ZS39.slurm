#!/bin/bash
#SBATCH -J ZS39
#SBATCH -p RTX4090-1
#SBATCH -N 1
#SBATCH -o ZS39.out
#SBATCH -e ZS39.err
python train_ZS_H3K9.py
