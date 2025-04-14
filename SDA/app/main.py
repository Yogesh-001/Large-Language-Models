from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

model_name = "meta-llama/Llama-3.2-1B"
base_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = PeftModel.from_pretrained(base_model, "./llama3_SDA_model/checkpoint-6")

def generate(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.9)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = input("Enter your prompt:\n")
    print("Generated Fix:\n", generate(prompt))
