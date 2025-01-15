#!/bin/bash
#SBATCH -J py_pdi
#SBATCH -p RTX3090-6
#SBATCH -N 1
#SBATCH -o py.out
#SBATCH -e py.err
python train.py
