from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
from transformers import default_data_collator
import torch

MODEL_NAME = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
dataset = load_from_disk("tokenized_debug_dataset")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # adjust if needed
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./llama3_SDA_model",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    gradient_accumulation_steps=2,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    data_collator=default_data_collator,
    # label_names=["labels"]
)

trainer.train()
