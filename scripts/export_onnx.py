"""Экспорт BGE-M3 в ONNX."""

import torch
import warnings

from transformers import AutoModel, AutoTokenizer

from model_wrapper import BGE_M3_Wrapper

warnings.filterwarnings("ignore", category=Warning)


def export_explicit_model(onnx_path="model.onnx"):
    model = AutoModel.from_pretrained("deepvk/USER-bge-m3")
    tokenizer = AutoTokenizer.from_pretrained("deepvk/USER-bge-m3")
    model.eval()

    wrapper = BGE_M3_Wrapper(model)
    wrapper.eval()

    vocab_size = len(tokenizer)
    dummy_ids = torch.randint(0, vocab_size, (1, 512), dtype=torch.long)
    dummy_mask = torch.ones((1, 512), dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        opset_version=18,
        do_constant_folding=True,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
    )

    print(f"Модель экспортирована в: {onnx_path}")
    return onnx_path
