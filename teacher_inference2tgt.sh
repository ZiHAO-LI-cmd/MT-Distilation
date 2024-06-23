#!/bin/bash


#SBATCH -A project_2001403
#SBATCH -J en-uk
#SBATCH -o en-uk.%j.out
#SBATCH -e en-uk.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chao.wang@helsinki.fi



echo "Starting at `date`"

set -e
# replace the following lines with specific configuration
/scratch/project_2010444/venv/bin/python ./teacher_inference2tgt.py \     # path to the python program and the inference script (depends on the specific task)
    --input_file=./source.en \     # path to the input file
    --output_file=./translated.tgt \  # path to the output file
    --model_path=./model \    # path to the model
    --tokenizer_path=./tokenizer \  # path to the tokenizer
    --prefix=">>tgt<< " \   # prefix for the target language, e.g., ">>ukr<< " for Ukrainian
    --batch_size=32   # batch size

echo "Finishing at `date`"
