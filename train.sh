#!/bin/bash

python train.py \
  --learning_rate 2e-5 \
  --batch_size 24 \
  --num_epochs 1 \
  --src_file_path "./data/en-uk/sampled.1m.en" \
  --tgt_file_path "./data/en-uk/sampled.1m.uk" \
  --tgt_lang "ukr" \
  --model_name "Helsinki-NLP/opus-mt-tc-big-en-zle" \
  --model_save_path "./models_1m_parallel" \
  --wandb_project "MT-Distilation"\
  --wandb_run "Test_1M_Parallel"
