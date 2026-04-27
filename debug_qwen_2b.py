from model import Qwen3_5MultimodalModel
import torch

qwen = Qwen3_5MultimodalModel(model_id="Qwen/Qwen3.5-2B")
print(f"Vision model hidden size: {qwen.hf_model.model.visual.embed_dim}")
