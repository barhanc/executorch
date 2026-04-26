import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from examples.models.qwen3_5.model import SimpleVisionWrapper

def compare_outputs():
    model_id = "Qwen/Qwen3.5-0.8B"
    hf_model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float32)
    
    # Create input
    pixel_values = torch.randn(1, 3, 512, 512)
    
    # 1. Eager/Official vision path
    # Qwen Vision typically uses: model.model.visual(x)
    grid_thw = torch.tensor([[1, 32, 32]], dtype=torch.int64)
    with torch.no_grad():
        orig_out = hf_model.model.visual(pixel_values, grid_thw)
        orig_hidden = orig_out.last_hidden_state
        print(f"Original vision model last_hidden_state shape: {orig_hidden.shape}")
        
    # 2. Wrapper path
    wrapper = SimpleVisionWrapper(hf_model.model.visual)
    with torch.no_grad():
        wrapper_out = wrapper(pixel_values)
        print(f"Wrapper output shape: {wrapper_out.shape}")
        
    if orig_hidden.shape != wrapper_out.shape:
        print("SHAPE MISMATCH DETECTED!")
    else:
        print("Shapes match.")

if __name__ == "__main__":
    compare_outputs()
