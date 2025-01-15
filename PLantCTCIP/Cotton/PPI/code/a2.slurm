#!/bin/bash
#SBATCH -J G-c-a2
#SBATCH -p RTX4090-1
#SBATCH -N 1
#SBATCH -o a2.out
#SBATCH -e a2.err
python train_a2.py
