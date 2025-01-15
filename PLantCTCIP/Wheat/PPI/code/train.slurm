#!/bin/bash
#SBATCH -J G-w
#SBATCH -N 1
#SBATCH -p RTX4090-1
#SBATCH -o w.out
#SBATCH -e w.err
python train.py
