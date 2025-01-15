#!/bin/bash
#SBATCH -J G-c-g3
#SBATCH -N 1
#SBATCH -o g3.out
#SBATCH -e g3.err
python get_3_data.py
