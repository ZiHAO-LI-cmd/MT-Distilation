import os
import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import log_softmax, softmax
from torch.utils.data import Dataset, DataLoader
import wandb
import argparse


class TranslationDataset(Dataset):
    def __init__(
        self, src_sentences, tgt_sentences, tgt_lang, tokenizer, max_length=512
    ):
        self.tokenizer = tokenizer
        self.src_sentences = [f" >>{tgt_lang}<< " + sent for sent in src_sentences]
        self.tgt_sentences = tgt_sentences
        self.max_length = max_length

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]

        model_inputs = self.tokenizer(
            text=src_text,
            text_pair=tgt_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        decoder_input_ids = self.tokenizer.encode(
            tgt_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).squeeze()

        model_inputs["decoder_input_ids"] = decoder_input_ids
        model_inputs["labels"] = decoder_input_ids.clone()

        model_inputs = {key: val.squeeze(0) for key, val in model_inputs.items()}

        return model_inputs


def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines()]
        return lines
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return []


def calculate_loss(
    student_outputs, teacher_outputs, labels, temperature, alpha, batch_size
):
    s_logits = student_outputs.logits
    t_logits = teacher_outputs.logits

    vocab_size = s_logits.size(-1)
    ce_logits = s_logits.view(-1, vocab_size)
    ce_labels = labels.view(-1)
    ce_loss = torch.nn.functional.cross_entropy(ce_logits, ce_labels)
    student_log_probs = log_softmax(s_logits.view(-1, vocab_size) / temperature, dim=-1)
    teacher_probs = softmax(t_logits.view(-1, vocab_size) / temperature, dim=-1)

    distill_loss = torch.nn.functional.kl_div(
        student_log_probs, teacher_probs, reduction="batchmean"
    )
    loss = (1 - alpha) * ce_loss + (
        alpha * temperature**2 / batch_size**2
    ) * distill_loss
    return loss


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            teacher_outputs = teacher_model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                decoder_input_ids=batch["decoder_input_ids"],
            )
            student_outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                decoder_input_ids=batch["decoder_input_ids"],
            )
            loss = calculate_loss(
                student_outputs,
                teacher_outputs,
                batch["labels"],
                temperature,
                alpha,
                batch_size,
            )
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


parser = argparse.ArgumentParser(
    description="Train a machine translation model with distillation."
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.001,
    help="Learning rate for the optimizer.",
)
parser.add_argument(
    "--batch_size", type=int, default=4, help="Batch size for training and validation."
)
parser.add_argument(
    "--num_epochs", type=int, default=3, help="Number of epochs to train."
)
parser.add_argument(
    "--temperature", type=float, default=5.0, help="Temperature for distillation."
)
parser.add_argument(
    "--alpha", type=float, default=0.5, help="Weight for the distillation loss."
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
    default="translation-distillation",
    help="Weights & Biases project name.",
)
args = parser.parse_args()

learning_rate = args.learning_rate
batch_size = args.batch_size
num_epochs = args.num_epochs
temperature = args.temperature
alpha = args.alpha
src_file_path = args.src_file_path
tgt_file_path = args.tgt_file_path
tgt_lang = args.tgt_lang
model_save_path = args.model_save_path
model_name = args.model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model = MarianMTModel.from_pretrained(model_name)
teacher_model.to(device)
teacher_model.eval()
tokenizer = MarianTokenizer.from_pretrained(model_name)

src_sentences = read_text_file(src_file_path)
tgt_sentences = read_text_file(tgt_file_path)

assert len(src_sentences) == len(
    tgt_sentences
), "The number of sentences must be the same in both files."

split_ratio = 0.9
split_index = int(len(src_sentences) * split_ratio)

# Training Set
train_src_sentences = src_sentences[:split_index]
train_tgt_sentences = tgt_sentences[:split_index]
train_dataset = TranslationDataset(
    train_src_sentences, train_tgt_sentences, tgt_lang, tokenizer
)

# Validation Set
val_src_sentences = src_sentences[split_index:]
val_tgt_sentences = tgt_sentences[split_index:]
val_dataset = TranslationDataset(
    val_src_sentences, val_tgt_sentences, tgt_lang, tokenizer
)

student_config = teacher_model.config
student_model = MarianMTModel(student_config)
student_model.to(device)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)


run = wandb.init(
    project=args.wandb_project,
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "temperature": temperature,
        "alpha": alpha,
    },
)

best_loss = float("inf")
for epoch in range(num_epochs):
    student_model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}

        teacher_outputs = teacher_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            decoder_input_ids=batch["decoder_input_ids"],
        )
        student_outputs = student_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            decoder_input_ids=batch["decoder_input_ids"],
        )

        loss = calculate_loss(
            student_outputs,
            teacher_outputs,
            batch["labels"],
            temperature,
            alpha,
            batch_size,
        )
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = validate(student_model, val_loader, device)
    wandb.log({"Training Loss": avg_train_loss, "Validation Loss": avg_val_loss})

    # Keep best checkpoint
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        model_save_dir = os.path.join(model_save_path, "best.pt")
        student_model.save_pretrained(model_save_dir)
        print(f"Saved best model checkpoint at {model_save_dir}")

model_save_dir = os.path.join(model_save_path, "last.pt")
student_model.save_pretrained(model_save_dir)
print(f"Saved last model at {model_save_dir}")
wandb.finish()
