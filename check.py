import tensorflow as tf
import numpy as np
import PIL
import requests
import cv2
import mediapipe as mp

# Load models
model1 = tf.keras.models.load_model('D:\\yoii\\GitHub\\model\\VGG16FineTune.keras')
model2 = tf.keras.models.load_model('D:\\yoii\\GitHub\\model\\VGG19FineTune.keras')

# Print model summaries
print("VGG16 Model Summary:")
model1.summary()

print("\nVGG19 Model Summary:")
model2.summary()

# Check versions of required dependencies
print(f"\nTensorFlow version: {tf.__version__}")
print(f"Numpy version: {np.__version__}")
print(f"Pillow version: {PIL.__version__}")
print(f"Requests version: {requests.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"MediaPipe version: {mp.__version__}")

# Optionally, if optimizer is used, print optimizer configurations
if model1.optimizer:
    optimizer_config = model1.optimizer.get_config()
    print(f"\nVGG16 Optimizer Configuration: {optimizer_config}")

if model2.optimizer:
    optimizer_config = model2.optimizer.get_config()
    print(f"\nVGG19 Optimizer Configuration: {optimizer_config}")

# Print model configurations
model1_config = model1.get_config()
print(f"\nVGG16 Model Configuration: {model1_config}")

model2_config = model2.get_config()
print(f"\nVGG19 Model Configuration: {model2_config}")
