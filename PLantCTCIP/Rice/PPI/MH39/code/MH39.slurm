#!/bin/bash
#SBATCH -J MH39
#SBATCH -p RTX4090-0
#SBATCH -N 1
#SBATCH -o MH39.out
#SBATCH -e MH39.err
python train_MH_H3K9.py
