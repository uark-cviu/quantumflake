import torch
from ultralytics import YOLO
import quantumflake.models.classifier as classifier

print("--- Starting Model Export to ONNX ---")
print("\n[1/2] Converting FlakeLayerClassifier...")

classifier_model = classifier.FlakeLayerClassifier(num_materials=2, material_dim=64)
classifier_weights = '/home/sankalp/quantumflake/weights/flake_monolayer_classifier.pth'
checkpoint = torch.load(classifier_weights)
classifier_model.load_state_dict(checkpoint['model_state_dict']) # Corrected line
classifier_model.eval()
batch_size = 1
pixel_dummy_input = torch.randn(batch_size, 3, 224, 224)
material_dummy_input = torch.tensor([0], dtype=torch.long)
classifier_dummy_inputs = (pixel_dummy_input, material_dummy_input)
classifier_onnx_path = "flake_classifier.onnx"
torch.onnx.export(classifier_model,
                  classifier_dummy_inputs,
                  classifier_onnx_path,
                  opset_version=12,
                  input_names=['pixel_values', 'material'],
                  output_names=['output'])

print(f"FlakeLayerClassifier saved to: {classifier_onnx_path}")

print("\n[2/2] Converting YOLO Detector...")

yolo_weights_path = '/home/sankalp/quantumflake/weights/uark_detector_v3.pt'
detector_model = YOLO(yolo_weights_path)
detector_onnx_path = detector_model.export(format='onnx', imgsz=640)

print(f"YOLO Detector saved to: {detector_onnx_path}")
print("\n--- All models exported successfully. ---")