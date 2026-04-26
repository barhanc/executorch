import torch
from transformers import AutoProcessor
from PIL import Image
import requests

model_id = "Qwen/Qwen3.5-0.8B"
processor = AutoProcessor.from_pretrained(model_id)
image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image = Image.open(requests.get(image_url, stream=True).raw)
resized_image = image.resize((512, 512))

inputs = processor(text="Describe this image.", images=resized_image, return_tensors="pt")
print(f"input_ids shape: {inputs['input_ids'].shape}")
print(f"pixel_values shape: {inputs['pixel_values'].shape}")
print(f"image_grid_thw shape: {inputs['image_grid_thw'].shape}")
print(f"image_grid_thw: {inputs['image_grid_thw']}")
