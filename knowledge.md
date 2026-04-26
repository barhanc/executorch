# Qwen 3.5 Multimodal Export to ExecuTorch

## Progress Summary

### 1. Architecture Analysis
- **Model**: Qwen3.5-0.8B is a native multimodal model.
- **Components**:
    - `visual`: `Qwen3_5VisionModel`.
    - `language_model`: `Qwen3_5TextModel`.
- **Vision Encoder Specifics**:
    - Uses `Conv3d` for patch embedding.
    - Contains data-dependent control flow in `fast_pos_embed_interpolate` (uses `torch.linspace` with values derived from input tensor `grid_thw`).
    - Uses `cu_seqlens` for Flash Attention / SDPA.

### 2. Implementation Details
- **`examples/models/qwen3_5/model.py`**:
    - Implemented `Qwen3_5` wrapper for the text transformer.
    - Implemented `SimpleVisionWrapper` to bypass data-dependent control flow by precomputing positional embeddings for a fixed image size.
    - Implemented `Qwen3_5MultimodalModel` (inherits from `EagerModelBase`) to coordinate the export of different components.
- **`examples/models/qwen3_5/export_qwen3_5_multimodal.py`**:
    - Created an export script that follows the LLaVA pattern (exporting `vision_encoder`, `token_embedding`, and `text_decoder` separately).
    - Uses `LLMEdgeManager` for text model export.
    - Uses `SimpleVisionWrapper` for vision encoder export.

### 3. Challenges & Resolutions
- **Custom Ops**: Encountered `AssertionError: Expected 1 library but got 0` for `sdpa_with_kv_cache`. 
    - *Resolution*: Disabled `use_sdpa_with_kv_cache_op` by default to use standard PyTorch ops.
- **Exporting Vision Encoder**: `torch.export.export` failed on `torch.linspace` in the interpolation logic.
    - *Resolution*: Created `SimpleVisionWrapper` which precomputes embeddings during initialization and uses them in `forward`, effectively making them constants for the export.
- **Import Errors**: Encountered `ImportError` due to `PYTHONPATH`.
    - *Resolution*: Run with `PYTHONPATH=..` from the `executorch` root.
- **Size Mismatch**: Encountered `RuntimeError: Error(s) in loading state_dict for Transformer`.
    - *Findings*: Qwen3.5-0.8B uses `dim=1024`, `head_dim=256`, and gated attention (`use_q_gate=True`).
    - *Resolution*: Updated `ModelArgs` initialization in `model.py` to correctly map `hidden_size -> dim`, `intermediate_size -> hidden_dim`, and pass `head_dim` and `attn_output_gate`.
- **Prefill Embedding Logic**: Initial logic was doubling tokens or using wrong placeholders.
    - *Findings*: `Qwen3VLProcessor` expands the single image token into a sequence of placeholders in `input_ids`.
    - *Resolution*: Updated `get_inputs_for_prefill` to return the full `input_ids`. Updated `export_all` to use `masked_scatter` to replace the embeddings of the image tokens with vision features.
- **Export Timeouts**: Exporting large models can take more than 5 minutes.
    - *Resolution*: Run export in background and reduce `max_seq_len` to 3072 (enough for the ~2770 tokens in the demo prefill).

- **Fixed Size Export**: User requested fixed size images (512x512).
    - *Implementation*: Updated `get_inputs_for_prefill` in `model.py` to resize images to 512x512.
- **Conv3d Support**: ExecuTorch's portable convolution kernel only supports 3D (1D conv) or 4D (2D conv) tensors. Qwen3.5's `Conv3d` takes 5D tensors.
    - *Resolution*: Implemented `SimplePatchEmbed` in `model.py` which replaces the `Conv3d` layer with an equivalent `Conv2d` layer by folding the temporal dimension into the channel dimension and reshaping the input accordingly.

- **Performance Optimization**: 
    - *Bottleneck*: Linear attention loops were being unrolled by Dynamo when using long example inputs.
    - *Resolution*: Used `example_seq_len = 1` for the text decoder export.
    - *Result*: Export time reduced to ~1 minute.
- **Verification**:
    - *Method*: Compared eager PyTorch output with ExecuTorch output on a real 512x512 image + prompt.
    - *Result*: **SUCCESS**. ExecuTorch outputs match eager model outputs. Sequential prefill was used for ET verification to handle the static-shape text decoder.

## Verification
Use the `test_multimodal_runner.py` script to verify the exported model with the multimodal pipeline. Do not use `verify_` scripts as they bypass the standard runner pipeline.

```bash
source .venv/bin/activate
PYTHONPATH=. python3 test_multimodal_runner.py
```

## Summary of Files
- `model.py`: Multimodal model definitions and `SimplePatchEmbed` (Conv3d workaround).
- `export_qwen3_5_multimodal.py`: Optimized export script.
- `verify_qwen3_5_multimodal.py`: Real-data verification script.

