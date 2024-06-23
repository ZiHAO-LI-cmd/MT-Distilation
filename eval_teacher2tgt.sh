#!/bin/bash

#SBATCH -A project_2001403
#SBATCH -J eval_teacher
#SBATCH -o ./slurm/eval_teacher.%j.out
#SBATCH -e ./slurm/eval_teacher.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chao.wang@helsinki.fi



echo "Starting at `date`"

set -e


/scratch/project_2010444/venv/bin/python ./inference_teacher.py \ # path to the python program and the inference script (depends on the specific task)
    --input_file=./eval.en \   # path to the input file
    --output_file=./teacher.bel \   # path to the output file
    --model_path=./model \   # path to the model
    --tokenizer_path=./tokenizer \  # path to the tokenizer
    --batch_size=32 \  # batch size
    --prefix=">>tgt<< " \   # prefix to be added to the input sentence 

echo "Finishing at `date`"
