# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from argparse import ArgumentParser

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.extension.llm.export.quantizer_lib import (
    get_pt2e_quantizers,
    PT2EQuantOptions,
    DynamicQuantLinearOptions,
)
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    replace_kv_cache_with_custom_kv_cache,
)
from executorch.examples.models.llama.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_weight_transform,
)
from model import Qwen3_5MultimodalModel, SimpleVisionWrapper
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)

from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import (
    ConstraintBasedSymShapeEvalPass,
)
from executorch.extension.llm.export.builder import DType, LLMEdgeManager
from torch.nn.attention import SDPBackend

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

class Qwen3_5EdgeManager(LLMEdgeManager):
    def export(self) -> "Qwen3_5EdgeManager":
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            self.export_program = torch.export.export(
                self.model,
                self.example_inputs,
                strict=False,
            )
            self.pre_autograd_graph_module = self.export_program.module()
        return self

def export_text_model(qwen, quantize=False):
    source_transforms = []
    if qwen.text_model_args.use_sdpa_with_kv_cache_op:
        source_transforms.append(replace_kv_cache_with_custom_kv_cache)
        source_transforms.append(replace_sdpa_with_custom_op)

    # Static 1-token export. 
    dim = qwen.text_model_args.dim
    example_embeddings = torch.randn(1, 1, dim)
    example_input_pos = torch.zeros(1, dtype=torch.int64)

    text_model_em = Qwen3_5EdgeManager(
        model=qwen,
        modelname="qwen_text_model",
        max_seq_len=qwen.text_model_args.max_seq_len,
        dtype=DType.fp32,
        use_kv_cache=True,
        example_inputs=(example_embeddings, example_input_pos),
    )
    
    if quantize:
        logging.info("Applying 8da4w weight transform...")
        quant_transform = get_quant_weight_transform(
            quantization_mode="8da4w",
            group_size=128,
            computation_dtype=DType.fp32,
        )
        source_transforms.append(quant_transform)
    
    manager = text_model_em.source_transform(source_transforms).export()
    
    if quantize:
        logging.info("Quantizing text_decoder with get_pt2e_quantizers...")
        quant_options = PT2EQuantOptions(
            quantize_linear=DynamicQuantLinearOptions(is_per_channel=True)
        )
        quantizers = get_pt2e_quantizers(quant_options)
        manager = manager.pt2e_quantize(quantizers)
        
    return manager.export_program

def export_image_encoder(qwen, quantize=False):
    vision_encoder = SimpleVisionWrapper(qwen.hf_model.model.visual)
    
    # Input is raw image [1, 3, 512, 512]
    example_input = torch.randn(1, 3, 512, 512)
    
    manager = Qwen3_5EdgeManager(
        model=vision_encoder,
        modelname="qwen_image_encoder",
        max_seq_len=2048,
        dtype=DType.fp32,
        use_kv_cache=False,
        example_inputs=(example_input,),
    )
    manager.export()

    if quantize:
        logging.info("Quantizing vision_encoder...")
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config())
        manager = manager.pt2e_quantize([quantizer])

    return manager.export_program

def export_token_embedding(qwen):
    example_ids = torch.zeros(1, 1, dtype=torch.int64)

    with torch.no_grad():
        token_embedding_ep = torch.export.export(
            qwen.hf_model.model.language_model.embed_tokens,
            (example_ids,),
            strict=False,
        )
    return token_embedding_ep

def export_all(qwen_model: Qwen3_5MultimodalModel, quantize=False):
    qwen = qwen_model.get_eager_model()

    logging.info("Exporting vision_encoder...")
    # Quantization only applied to text model, vision encoder remains FP32
    image_encoder_ep = export_image_encoder(qwen, quantize=False)

    logging.info("Exporting text_decoder...")
    text_model_ep = export_text_model(qwen, quantize=quantize)

    logging.info("Exporting token_embedding...")
    token_embedding_ep = export_token_embedding(qwen)

    logging.info("Lowering and combining...")
    lowered_and_edge = to_edge_transform_and_lower(
        {
            "vision_encoder": image_encoder_ep,
            "token_embedding": token_embedding_ep,
            "text_decoder": text_model_ep,
        },
        partitioner={
            "vision_encoder": [XnnpackPartitioner()],
            "text_decoder": [XnnpackPartitioner()],
        },
        constant_methods={
            "get_max_seq_len": qwen_model.max_seq_len,
            "get_max_context_len": qwen_model.max_context_len,
            "get_bos_id": qwen.hf_model.config.text_config.bos_token_id or 248045,
            "get_eos_ids": [qwen.hf_model.config.text_config.eos_token_id or 248044],
            "enable_dynamic_shape": False,
        },
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    executorch_program = lowered_and_edge.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            passes=[QuantFusionPass()] if quantize else [],
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass={
                "vision_encoder": ConstraintBasedSymShapeEvalPass(),
                "text_decoder": ConstraintBasedSymShapeEvalPass(),
                "token_embedding": ConstraintBasedSymShapeEvalPass(),
            },
        )
    )
    return executorch_program

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max-seq-len", default=1024, type=int)
    parser.add_argument("--pte-name", default="qwen3_5_v.pte")
    parser.add_argument("--quantize", action="store_true", help="Quantize the model to 8-bit weights.")
    args = parser.parse_args()

    qwen_model = Qwen3_5MultimodalModel(
        model_id=args.model_id,
        max_seq_len=args.max_seq_len,
        max_context_len=args.max_seq_len,
    )

    executorch_program = export_all(qwen_model, quantize=args.quantize)

    with open(args.pte_name, "wb") as f:
        executorch_program.write_to_file(f)
    logging.info(f"Exported ExecuTorch program to {args.pte_name}")

if __name__ == "__main__":
    main()
