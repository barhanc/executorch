import logging
import torch
from executorch.examples.models.qwen3_5.model import Qwen3_5MultimodalModel, SimpleVisionWrapper
from executorch.extension.llm.export.builder import DType, LLMEdgeManager

logging.basicConfig(level=logging.INFO)

print("Loading model...")
qwen_model = Qwen3_5MultimodalModel()
qwen = qwen_model.get_eager_model()

print("Getting prefill inputs...")
input_ids, pixel_values, grid_thw = qwen_model.get_inputs_for_prefill()

print("Exporting vision encoder...")
vision_encoder = SimpleVisionWrapper(qwen.hf_model.model.visual, grid_thw)

with torch.no_grad():
    exp = torch.export.export(vision_encoder, (pixel_values,), strict=False)
print("Export vision encoder successful!")

# Try to save it to see if it finishes
from executorch.exir import to_edge
edge = to_edge(exp)
print("To edge successful!")
