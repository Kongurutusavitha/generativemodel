

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Text generation function
def generate_text(prompt, max_len=200, temp=0.7, top_p=0.9, top_k=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs, max_length=max_len, temperature=temp,
        top_p=top_p, top_k=top_k, do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example prompts
topics = [
    "The importance of renewable energy",
    "Advancements in artificial intelligence",
    "The future of space exploration"
]

for topic in topics:
    print(f"Prompt: {topic}\n{generate_text(topic)}\n{'-'*80}\n")
