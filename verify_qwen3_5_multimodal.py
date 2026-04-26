import torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer
from examples.models.qwen3_5.model import Qwen3_5MultimodalModel
from PIL import Image
import requests
import time

def main():
    model_id = "Qwen/Qwen3.5-0.8B"
    pte_path = "qwen3_5_v_q4.pte"
    print(f"Loading model {model_id} and processor...")
    qwen_model = Qwen3_5MultimodalModel(max_seq_len=512, max_context_len=512, model_id=model_id)
    eager_qwen = qwen_model.get_eager_model()
    eager_qwen.eval()
    
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    print(f"Downloading image from {image_url}...")
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    
    prompt = "cat"
    print(f"Prompt: {prompt}")
    
    # Process inputs
    image = image.resize((512, 512))
    inputs = qwen_model.processor(
        text=qwen_model.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True
        ),
        images=image,
        return_tensors="pt",
    )
    
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]
    print(f"DEBUG: pixel_values.shape: {pixel_values.shape}")
    
    print(f"Input tokens: {input_ids.shape[1]}")

    print("\n--- Running ExecuTorch ---")
    with open("qwen3_5_v.pte", "rb") as f:
        pte_buffer = f.read()
    et_module = _load_for_executorch_from_buffer(pte_buffer)
pte_path = "qwen3_5_v.pte"
...
# vision_encoder
# The vision encoder expects the pre-processed pixel values shaped as [1, 3, 512, 512]
# to match the static shape defined in the export script.
formatted_pixel_values = pixel_values.view(1024, 3, 512).permute(1, 0, 2).reshape(1, 3, 512, 512).contiguous()

et_image_embeds = et_module.run_method("vision_encoder", (formatted_pixel_values,))[0]
    # token_embedding (sequential)
    et_token_embeds_list = []
    for i in range(input_ids.shape[1]):
        current_id = input_ids[:, i:i+1]
        et_token_embeds_list.append(et_module.run_method("token_embedding", (current_id,))[0])
    et_token_embeds = torch.cat(et_token_embeds_list, dim=1)
    
    # Merge
    et_image_token_mask = (input_ids == qwen_model.model.config.image_token_id).unsqueeze(-1).expand_as(et_token_embeds)
    et_embeddings = et_token_embeds.masked_scatter(et_image_token_mask, et_image_embeds)

    # Prefill (sequential)
    print(f"Starting sequential prefill for {et_embeddings.shape[1]} tokens...")
    start_time = time.time()
    et_logits = None
    for i in range(et_embeddings.shape[1]):
        current_emb = et_embeddings[:, i:i+1, :]
        input_pos = torch.tensor([i], dtype=torch.int64)
        et_logits = et_module.run_method("text_decoder", (current_emb, input_pos))[0]
    print(f"ET Prefill finished in {time.time() - start_time:.2f}s")

    # Generation
    max_new_tokens = 50
    curr_pos = et_embeddings.shape[1]
    if et_logits.ndim == 3:
        next_token = torch.argmax(et_logits[:, -1, :], dim=-1)
    else:
        next_token = torch.argmax(et_logits[-1, :], dim=-1)
    
    print(f"ET Output: ", end="", flush=True)
    print(qwen_model.processor.tokenizer.decode(next_token), end="", flush=True)

    for _ in range(max_new_tokens - 1):
        if next_token.item() in [qwen_model.model.config.text_config.eos_token_id]:
            break
        next_token_2d = next_token.reshape(1, 1)
        next_emb = et_module.run_method("token_embedding", (next_token_2d,))[0]
        et_logits = et_module.run_method("text_decoder", (next_emb, torch.tensor([curr_pos], dtype=torch.int64)))[0]
        if et_logits.ndim == 3:
            next_token = torch.argmax(et_logits[:, -1, :], dim=-1)
        else:
            next_token = torch.argmax(et_logits[-1, :], dim=-1)
        print(qwen_model.processor.tokenizer.decode(next_token), end="", flush=True)
        curr_pos += 1
    print("\n")

    print("--- Running Eager ---")
    with torch.no_grad():
        eager_image_outputs = eager_qwen.image_embedding(pixel_values, grid_thw)
        if isinstance(eager_image_outputs, torch.Tensor):
            eager_image_embeds = eager_image_outputs
        else:
            eager_image_embeds = eager_image_outputs.pooler_output
        
        eager_token_embeds = eager_qwen.embed_tokens(input_ids)
        eager_image_token_mask = (input_ids == qwen_model.model.config.image_token_id).unsqueeze(-1).expand_as(eager_token_embeds)
        eager_embeddings = eager_token_embeds.masked_scatter(eager_image_token_mask, eager_image_embeds)
        
        start_time = time.time()
        eager_input_pos = torch.arange(eager_embeddings.shape[1], dtype=torch.int64)
        eager_logits = eager_qwen.text_model(None, {"input_pos": eager_input_pos}, eager_embeddings)
        print(f"Eager Prefill finished in {time.time() - start_time:.2f}s")
        
        if eager_logits.ndim == 3:
            next_token = torch.argmax(eager_logits[:, -1, :], dim=-1)
        else:
            next_token = torch.argmax(eager_logits[-1, :], dim=-1)
            
        print(f"Eager Output: ", end="", flush=True)
        print(qwen_model.processor.tokenizer.decode(next_token), end="", flush=True)
        
        eager_curr_pos = eager_embeddings.shape[1]
        for _ in range(max_new_tokens - 1):
            if next_token.item() in [qwen_model.model.config.text_config.eos_token_id]:
                break
            next_emb = eager_qwen.embed_tokens(next_token.reshape(1, 1))
            eager_logits = eager_qwen.text_model(None, {"input_pos": torch.tensor([eager_curr_pos], dtype=torch.int64)}, next_emb)
            if eager_logits.ndim == 3:
                next_token = torch.argmax(eager_logits[:, -1, :], dim=-1)
            else:
                next_token = torch.argmax(eager_logits[-1, :], dim=-1)
            print(qwen_model.processor.tokenizer.decode(next_token), end="", flush=True)
            eager_curr_pos += 1
    print("\n")

if __name__ == "__main__":
    main()
