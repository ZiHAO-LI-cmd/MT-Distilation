import argparse
import os
import torch
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def load_model_tokenizer(model_path, tokenizer_path):
    model = MarianMTModel.from_pretrained(model_path)
    tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

class InferenceDataset(Dataset):
    def __init__(self, source_texts, tokenizer, max_length=512):
        self.source_texts = source_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        source_encoding = self.tokenizer(
            source_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
        }

def create_inference_dataloader(source_texts, tokenizer, batch_size=32, max_length=512):
    dataset = InferenceDataset(source_texts, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def translate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, tokenizer = load_model_tokenizer(args.model_path, args.tokenizer_path)
    model.to(device)
    model.eval()

    with open(args.input_file, "r") as file:
        source_texts = file.readlines()

    inference_dataloader = create_inference_dataloader(source_texts, tokenizer, batch_size=args.batch_size)
    translated_texts = []

    for batch in tqdm(inference_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad(): # use beam search for inference 
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            batch_translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            translated_texts.extend(batch_translated_texts)

    with open(args.output_file, "w") as file:
        file.write("\n".join(translated_texts) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer')
    parser.add_argument('--input_file', type=str, required=True, help='Input file with sentences to translate')
    parser.add_argument('--output_file', type=str, required=True, help='Output file for translations')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()

    translate(args)

if __name__ == "__main__":
    main()
