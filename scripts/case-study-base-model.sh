#!/bin/bash
#SBATCH --job-name=case_study_base_model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:01:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1

# Setup env
export PYTHONPATH=../src

# Run experiment
python -m experiments.case_study_base_model
