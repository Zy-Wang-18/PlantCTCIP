#!/bin/bash
#SBATCH -J shoot
#SBATCH -p RTX4090-2
#SBATCH -N 1
#SBATCH -o shoot.out
#SBATCH -e shoot.err
python train.py
