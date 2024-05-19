#!/bin/bash

python distill.py \
  --learning_rate 0.001 \
  --batch_size 4 \
  --num_epochs 3 \
  --temperature 5.0 \
  --alpha 0.5 \
  --src_file_path "./data/en-uk/NLLB.en-uk.en" \
  --tgt_file_path "./data/en-uk/NLLB.en-uk.uk" \
  --tgt_lang "ukr" \
  --model_name "Helsinki-NLP/opus-mt-tc-big-en-zle" \
  --model_save_path "./models" \
  --wandb_project "MT-Distilation"
