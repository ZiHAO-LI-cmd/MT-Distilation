import torch
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import DataLoader
import sacrebleu

# Load model and tokenizer
model_path = "path/to/your/best/model"  # Modify this path to where best.pt is located
model = MarianMTModel.from_pretrained(model_path)
tokenizer = MarianTokenizer.from_pretrained(model_name)
model.to(device)
model.eval()

# Prepare the DataLoader for the test dataset
# This DataLoader should provide the source sentences and the target (reference) sentences
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Generate translations and evaluate them
all_predictions = []
all_references = []

with torch.no_grad():
    for batch in test_loader:
        src_texts = batch['src_texts']  # Ensure 'src_texts' are properly formatted input sentences
        tgt_texts = batch['tgt_texts']  # These are the reference translations

        # Tokenize the source texts
        inputs = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Generate translations
        translated = model.generate(**inputs)
        translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

        all_predictions.extend(translated_texts)
        all_references.extend([tgt_texts])  # sacrebleu expects a list of lists for references

# Calculate BLEU Score
bleu_score = sacrebleu.corpus_bleu(all_predictions, all_references)
print(f"BLEU score: {bleu_score.score}")

