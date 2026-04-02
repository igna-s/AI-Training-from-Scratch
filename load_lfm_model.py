#!/usr/bin/env python3
"""Load and run LFM2.5-350M model"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Path to local model
model_path = "models/LFM2.5-350M/snapshots/70810220513bfdbdfcbeade479f358390af187b4"

print("Loading LFM2.5-350M model...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model for CPU
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("[OK] Model loaded successfully on CPU")
print(f"Model type: {type(model)}")
print(f"Model config: {model.config}")

# Example inference
if __name__ == "__main__":
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=50,
            num_return_sequences=1,
            temperature=0.7
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
