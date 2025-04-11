from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
model_name = "meta-llama/Llama-3.2-1B"

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "./llama3_SDA_model/checkpoint-6")

# Set model to eval mode
model.eval()

prompt = """### Repository:
scikit-learn/scikit-learn

### Issue:
Fix index overflow in polynomial expansion by using int32

### Buggy Code Patch:
- row = [0]
- col = [n_features - 1]

### Fix the Code:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=80,
        temperature=1.15,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("🔧 Output:\n", generated[len(prompt):].strip())