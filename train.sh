#!/bin/bash

python train.py \
  --learning_rate 2e-5 \
  --batch_size 32 \
  --num_epochs 5 \
  --src_file_path "./data/en-uk/sampled.en" \
  --tgt_file_path "./data/en-uk/sampled.uk" \
  --tgt_lang "ukr" \
  --model_name "Helsinki-NLP/opus-mt-tc-big-en-zle" \
  --model_save_path "./models_1m_parallel" \
  --wandb_project "MT-Distilation"\
  --wandb_run "Test"
