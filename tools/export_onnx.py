import argparse
import os
from ultralytics import YOLOv10

parser = argparse.ArgumentParser(description="Export YOLOv10 model to ONNX")
parser.add_argument("--model", required=True, help="Path to .pt weight")
parser.add_argument("--output", default="model.onnx", help="Output onnx file")
args = parser.parse_args()

model = YOLOv10(args.model)
onnx_path = model.export(format="onnx", opset=13, simplify=True)
final_path = os.path.abspath(args.output)
os.rename(onnx_path, final_path)
print(f"ONNX model saved to {final_path}")
