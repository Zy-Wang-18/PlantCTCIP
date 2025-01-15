#!/bin/bash
#SBATCH -J in_tassel
#SBATCH -p RTX4090-2
#SBATCH -N 1
#SBATCH -o in_tassel.out
#SBATCH -e in_tassel.err
python train.py
