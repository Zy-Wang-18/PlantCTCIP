#!/bin/bash
#SBATCH -J shoot_pdi
#SBATCH -p RTX3090-6
#SBATCH -N 1
#SBATCH -o shoot.out
#SBATCH -e shoot.err
python train.py
