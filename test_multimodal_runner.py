import torch
from executorch.extension.llm.runner import MultimodalRunner, GenerationConfig, make_image_input, make_text_input
from transformers import AutoProcessor
from PIL import Image
import requests
import logging
import os
import numpy as np

logging.basicConfig(level=logging.INFO)

def main():
    # 1. Paths
    pte_path = "qwen3_5_v.pte"
    tokenizer_path = "tokenizer.json"
    model_id = "Qwen/Qwen3.5-0.8B"
    
    if not os.path.exists(pte_path):
        print(f"Error: {pte_path} not found.")
        return

    # 2. Initialize MultimodalRunner
    print(f"Initializing MultimodalRunner with {pte_path}...")
    try:
        runner = MultimodalRunner(pte_path, tokenizer_path)
    except Exception as e:
        print(f"Failed to init runner: {e}")
        return

    # 3. Prepare Image
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    print(f"Downloading image from {image_url}...")
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    resized_image = image.resize((512, 512))
    
    # Vision encoder expects [1, 3, 512, 512] float32 pixels in [0, 255]
    pixel_tensor = torch.from_numpy(np.array(resized_image)).permute(2, 0, 1).float().contiguous()
    
    # 4. Configure inputs
    # Since text_decoder is static [1, 1], we must pass 1 token at a time.
    # MultimodalRunner might not support this easily if it tokenizes the string internally.
    # We'll try to use make_text_input with very small strings or individual tokens.
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    def string_to_inputs(text):
        ids = tokenizer.encode(text)
        inputs = []
        for i in ids:
            # Re-decode each ID to a string to pass to make_text_input
            # or just use make_text_input with the string representation
            s = tokenizer.decode([i])
            inputs.append(make_text_input(s))
        return inputs

    inputs = []
    inputs.extend(string_to_inputs("<|im_start|>user\n<|vision_start|>"))
    
    # # Image tokens: 1024 tokens. 
    # # If the runner doesn't loop over image tokens, this will fail.
    inputs.append(make_image_input(pixel_tensor))
    
    inputs.extend(string_to_inputs("<|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"))
    
    # 4. Configure generation
    config = GenerationConfig(
        max_new_tokens=200,
        temperature=0.0,
        echo=False
    )
    
    # 5. Generate
    print("\n--- Generating ---")
    try:
        runner.generate(inputs, config)
    except Exception as e:
        print(f"\nGeneration failed: {e}")

    print("\n--- Done ---")

if __name__ == "__main__":
    main()
