import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import requests

model_id = "Qwen/Qwen3.5-0.8B"
model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float32)
processor = AutoProcessor.from_pretrained(model_id)

image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image = Image.open(requests.get(image_url, stream=True).raw)
inputs = processor(text="Describe this image.", images=image, return_tensors="pt")
print(f"Inputs keys: {inputs.keys()}")
pixel_values = inputs["pixel_values"]
grid_thw = inputs["image_grid_thw"]

print(f"Pixel values shape: {pixel_values.shape}")
print(f"grid_thw shape: {grid_thw.shape}")

class SimpleVisionWrapper(torch.nn.Module):
    def __init__(self, visual, grid_thw):
        super().__init__()
        self.patch_embed = visual.patch_embed
        self.blocks = visual.blocks
        self.merger = visual.merger
        
        # Precompute problematic parts
        with torch.no_grad():
            self.register_buffer("pos_embeds", visual.fast_pos_embed_interpolate(grid_thw))
            rotary_pos_emb = visual.rot_pos_emb(grid_thw)
            seq_len = self.pos_embeds.size(0)
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            self.register_buffer("rotary_pos_emb_buf", rotary_pos_emb)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            self.register_buffer("pos_emb_cos", emb.cos())
            self.register_buffer("pos_emb_sin", emb.sin())
            
            # cu_seqlens calculation for single image
            n_tokens = grid_thw[0, 1] * grid_thw[0, 2]
            cu_seqlens = torch.tensor([0, n_tokens], dtype=torch.int32)
            self.register_buffer("cu_seqlens", cu_seqlens)

    def forward(self, pixel_values):
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + self.pos_embeds
        
        position_embeddings = (self.pos_emb_cos, self.pos_emb_sin)
        
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                rotary_pos_emb=self.rotary_pos_emb_buf,
                position_embeddings=position_embeddings,
            )
        return self.merger(hidden_states)

wrapper = SimpleVisionWrapper(model.model.visual, grid_thw)
wrapper.eval()

try:
    with torch.no_grad():
        exp = torch.export.export(wrapper, (pixel_values,), strict=False)
    print("Export successful!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Export failed: {e}")
