import torch
from transformers import AutoModelForImageTextToText

model_id = "Qwen/Qwen3.5-0.8B"
model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
print(model.model.visual)
