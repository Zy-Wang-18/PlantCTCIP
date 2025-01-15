#!/bin/bash
#SBATCH -J process
#SBATCH -p RTX4090-2
#SBATCH -N 1
#SBATCH -o process.out
#SBATCH -e process.err
python process.py
