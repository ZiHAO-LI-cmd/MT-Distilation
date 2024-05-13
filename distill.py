import os
import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import log_softmax, softmax
from torch.utils.data import Dataset, DataLoader


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
    return lines


class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.src_sentences = [" >>ukr<< " + sent for sent in src_sentences]
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


def calculate_loss(student_outputs, teacher_outputs, labels):
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


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Helsinki-NLP/opus-mt-en-mul"
teacher_model = MarianMTModel.from_pretrained(model_name)
teacher_model.to(device)
teacher_model.eval()

tokenizer = MarianTokenizer.from_pretrained(model_name)

uk_file_path = "./data/en-uk/NLLB.en-uk.uk"
en_file_path = "./data/en-uk/NLLB.en-uk.en"

uk_sentences = read_text_file(uk_file_path)
en_sentences = read_text_file(en_file_path)

assert len(uk_sentences) == len(
    en_sentences
), "The number of sentences must be the same in both files."

train_dataset = TranslationDataset(en_sentences, uk_sentences, tokenizer)

student_config = teacher_model.config
student_model = MarianMTModel(student_config)
student_model.to(device)

learning_rate = 0.001
batch_size = 4
num_epochs = 3
temperature = 5
alpha = 0.5

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    student_model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
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

        loss = calculate_loss(student_outputs, teacher_outputs, batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Average loss: {total_loss / len(train_loader)}")

student_model.save_pretrained("distilled-opus-mt-translation-model")
