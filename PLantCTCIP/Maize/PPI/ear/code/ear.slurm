#!/bin/bash
#SBATCH -J ear
#SBATCH -p RTX3090-6
#SBATCH -N 1
#SBATCH -o ear.out
#SBATCH -e ear.err
python train.py
