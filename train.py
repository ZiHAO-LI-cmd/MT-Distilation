import os
import torch
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    MarianConfig,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import wandb
from transformers import get_scheduler
import argparse


def load_parallel_texts(src_path, tgt_path):
    try:
        with open(src_path, "r", encoding="utf-8") as src_file, open(
            tgt_path, "r", encoding="utf-8"
        ) as tgt_file:
            src_texts = [line.strip() for line in src_file.readlines()]
            tgt_texts = [line.strip() for line in tgt_file.readlines()]
    except FileNotFoundError:
        print(f"Error: The file {src_path} or {tgt_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred while reading files: {e}")
        return None

    assert len(src_texts) == len(
        tgt_texts
    ), "The number of lines in both files must be the same"

    return {"src": src_texts, "tgt": tgt_texts}


def preprocess_function(examples):
    try:
        inputs = examples["src"]
        targets = examples["tgt"]

        model_inputs = tokenizer(
            inputs, max_length=512, truncation=True, padding="max_length"
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=512, truncation=True, padding="max_length"
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    except KeyError as e:
        print(f"Key error in dataset: {e}. Check dataset keys and formatting.")
        return None
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return None


parser = argparse.ArgumentParser(
    description="Train a machine translation model with distillation."
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=2e-5,
    help="Learning rate for the optimizer.",
)
parser.add_argument(
    "--batch_size", type=int, default=4, help="Batch size for training and validation."
)
parser.add_argument(
    "--num_epochs", type=int, default=3, help="Number of epochs to train."
)

parser.add_argument(
    "--src_file_path",
    type=str,
    required=True,
    help="Source file path for training data.",
)
parser.add_argument(
    "--tgt_file_path",
    type=str,
    required=True,
    help="Target file path for training data.",
)
parser.add_argument("--tgt_lang", type=str, required=True, help="Target language code.")
parser.add_argument("--model_name", type=str, required=True, help="Teacher model name.")
parser.add_argument(
    "--model_save_path",
    type=str,
    default="./models",
    help="Path to save the trained model.",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="MT-Distilation",
    help="wandb project name",
)
parser.add_argument(
    "--wandb_run",
    type=str,
    default="test run",
    help="wandb run name.",
)
args = parser.parse_args()

learning_rate = args.learning_rate
batch_size = args.batch_size
num_epochs = args.num_epochs
src_file_path = args.src_file_path
tgt_file_path = args.tgt_file_path
tgt_lang = args.tgt_lang
model_save_path = args.model_save_path
model_name = args.model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_pairs = load_parallel_texts(src_file_path, tgt_file_path)
dataset = Dataset.from_dict(data_pairs)
tokenizer = MarianTokenizer.from_pretrained(model_name)
tokenized_datasets = dataset.map(preprocess_function, batched=True)
train_test_split = tokenized_datasets.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# config = MarianConfig.from_pretrained(model_name)
# student_model = MarianMTModel(config)

teacher_config = MarianConfig.from_pretrained(model_name)
config_dict = teacher_config.to_dict()
config_dict["num_hidden_layers"] = 3
config_dict["d_model"] = 512
config_dict["decoder_attention_heads"] = 8
config_dict["encoder_attention_heads"] = 8
config_dict["decoder_ffn_dim"] = 2048
config_dict["encoder_ffn_dim"] = 2048
student_config = MarianConfig(**config_dict)
student_model = MarianMTModel(student_config)
student_model.to(device)


training_args = TrainingArguments(
    output_dir=model_save_path,
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_dir="./log",
    logging_steps=1000,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
)


optimizer = torch.optim.AdamW(
    student_model.parameters(), lr=training_args.learning_rate
)
lr_scheduler = get_scheduler(
    name="linear",  #  "cosine", "cosine_with_restarts"
    optimizer=optimizer,
    num_warmup_steps=1000,
    num_training_steps=len(train_dataset) * training_args.num_train_epochs,
)

run = wandb.init(
    project=args.wandb_project,
    name=args.wandb_run,
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
    },
)

trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(optimizer, lr_scheduler)
)

trainer.train()
student_model = trainer.model
model_save_dir = os.path.join(model_save_path, "best_model")
student_model.save_pretrained(model_save_dir)
wandb.finish()
os.system("/usr/bin/shutdown")
