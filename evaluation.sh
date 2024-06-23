#!/bin/bash

#SBATCH -A project_2001403
#SBATCH -J eval
#SBATCH -o ./slurm/eval.%j.out
#SBATCH -e ./slurm/eval.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chao.wang@helsinki.fi



echo "Starting at `date`"

set -e


/scratch/project_2010444/venv/bin/python ./evaluation.py \
    --teacher_predictions ./teacher.en \    # path to the teacher predictions
    --student_predictions ./student.en \    # path to the student predictions
    --references ./eval.en \     # path to the reference file
    --sources ./eval.tgt  # path to the source file

echo "Finishing at `date`"
