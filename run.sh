#!/bin/bash

python distill.py \
  --learning_rate 0.001 \
  --batch_size 4 \
  --num_epochs 10 \
  --temperature 5.0 \
  --alpha 0.5 \
  --src_file_path "./data/en-uk/sampled.10k.en" \
  --tgt_file_path "./data/en-uk/sampled.10k.uk" \
  --tgt_lang "ukr" \
  --model_name "Helsinki-NLP/opus-mt-tc-big-en-zle" \
  --model_save_path "./models_v2" \
  --wandb_project "MT-Distilation"
