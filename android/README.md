# FRDC Road Damage Detection Android App

This directory contains a minimal Android application that demonstrates how to run the road damage detection model on a mobile device using ONNX Runtime.

## Prerequisites
- Android Studio Bumblebee or newer
- Android SDK and NDK installed
- Android device or emulator running Android 8.0+

## Exporting the Model
Export the YOLOv10 model to ONNX before building the app:

```bash
# Install ultralytics if not already installed
pip install ultralytics

# Export to onnx
yolo export model=<path_to_your_weight>.pt format=onnx opset=13 simplify
```

Copy the resulting `*.onnx` file to `app/src/main/assets/model.onnx`.

## Building
Open the `android` directory in Android Studio and build the `app` module. The application uses CameraX to capture frames and runs inference with ONNX Runtime. Post-processing and drawing of bounding boxes should be implemented in `MainActivity.kt`.

## Functionality
- Capture live camera frames using CameraX
- Run the ONNX model on each frame
- (TODO) Draw detection boxes on the preview

## Notes
This example provides a starting point. For better performance and full features (reading images/videos from storage, saving results), additional development is required.
