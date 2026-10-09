import torch
from transformers import AutoTokenizer, AutoConfig
from .train import AfriBertaForSentiment
import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_to_onnx(model_path, output_path="models/onnx/model.onnx"):
    print(f"Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path)
    model = AfriBertaForSentiment.from_pretrained(model_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # Dummy input
    text = "Hii ni mock text ya kutest export"
    inputs = tokenizer(
        text, return_tensors="pt", max_length=128, padding="max_length", truncation=True
    )

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits", "intensity"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "intensity": {0: "batch_size"},
        },
        opset_version=14,
    )
    print(f"ONNX model exported to {output_path}")

    # Quantization
    quantized_model_path = output_path.replace(".onnx", "_quantized.onnx")
    print("Quantizing model...")
    quantize_dynamic(output_path, quantized_model_path, weight_type=QuantType.QUInt8)
    print(f"Quantized model saved to {quantized_model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned PyTorch model directory",
    )
    parser.add_argument("--output_path", type=str, default="models/onnx/model.onnx")
    args = parser.parse_args()

    export_to_onnx(args.model_path, args.output_path)
