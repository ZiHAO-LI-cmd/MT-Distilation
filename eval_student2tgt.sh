#!/bin/bash

#SBATCH -A project_2001403
#SBATCH -J eval_en_be
#SBATCH -o ./slurm/eval_en_be.%j.out
#SBATCH -e ./slurm/eval_en_be.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chao.wang@helsinki.fi



echo "Starting at `date`"

set -e


/scratch/project_2010444/venv/bin/python ./inference_student.py \  # path to the python program and the inference script (depends on the specific task)
    --input_file=./eval.en \    # path to the input file
    --output_file=./student.tgt \   # path to the output file
    --model_path=./best_model \  # path to the model
    --tokenizer_path=./tokenizer \  # path to the tokenizer
    --batch_size=32 # batch size

echo "Finishing at `date`"
