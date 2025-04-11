from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
TOKENIZER.pad_token = TOKENIZER.eos_token

dataset = load_dataset("json", data_files="processed_dataset.jsonl", split="train")

def format_example(example):
    return {
        "text": example["prompt"] + "\n" + example["completion"]
    }

dataset = dataset.map(format_example)

# Tokenization
def tokenize(example):
    tokens = TOKENIZER(example["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, batched=True)

tokenized_dataset.save_to_disk("tokenized_debug_dataset")
