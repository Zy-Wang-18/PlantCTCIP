#!/bin/bash
#SBATCH -J py
#SBATCH -p RTX4090-2
#SBATCH -N 1
#SBATCH -o py.out
#SBATCH -e py.err
python train.py
