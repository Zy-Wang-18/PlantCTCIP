#!/bin/bash
#SBATCH -J in_ear
#SBATCH -p RTX4090-2
#SBATCH -N 1
#SBATCH -o in_ear.out
#SBATCH -e in_ear.err
python train.py
