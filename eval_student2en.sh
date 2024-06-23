#!/bin/bash

#SBATCH -A project_2001403
#SBATCH -J eval_be_en
#SBATCH -o ./slurm/eval_be_en.%j.out
#SBATCH -e ./slurm/eval_be_en.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chao.wang@helsinki.fi



echo "Starting at `date`"

set -e


/scratch/project_2010444/venv/bin/python ./inference_student.py \   # path to the python program and the inference script (depends on the specific task)
    --input_file=./eval.tgt \   # path to the input file
    --output_file=./student.en \    # path to the output file
    --model_path=./model \  # path to the model
    --tokenizer_path=./tokenizer \  # path to the tokenizer
    --batch_size=32 # batch size

echo "Finishing at `date`"
