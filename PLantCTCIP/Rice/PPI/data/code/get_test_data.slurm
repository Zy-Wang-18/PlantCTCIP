#!/bin/bash
#SBATCH -J get_test_data
#SBATCH -p RTX4090-2
#SBATCH -N 1
#SBATCH -o get_test_data.out
#SBATCH -e get_test_data.err
python get_test_data.py
