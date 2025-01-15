#!/bin/bash
#SBATCH -J ZSP2
#SBATCH -p RTX4090-2
#SBATCH -N 1
#SBATCH -o ZSP2.out
#SBATCH -e ZSP2.err
python train_ZS_RNAP2.py
