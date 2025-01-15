#!/bin/bash
#SBATCH -J G-gs
#SBATCH -N 1
#SBATCH -o gs.out
#SBATCH -e gs.err
python get_seq.py
