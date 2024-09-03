import tensorflow as tf
import numpy as np

# Path to the Keras model (.h5 or .keras)
model_path = 'D:/yoii/GitHub/model/VGG19FineTune.keras'
tflite_model_path = 'D:/yoii/GitHub/model/VGG19FineTune_int8.tflite'

# Load the Keras model
model = tf.keras.models.load_model(model_path)

# Print the model input shape for verification
print(f"Model input shape: {model.input_shape}")

# Ensure representative dataset matches the model's expected input shape
def representative_dataset_generator():
    for _ in range(100):
        data = np.random.rand(1, 224, 224, 3)  # Replace 224, 224, 3 with model's input size
        yield [data.astype(np.float32)]

# Convert the model with full quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_generator
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.int8
converter.inference_output_type = tf.uint8  # or tf.int8
tflite_model = converter.convert()

# Save the TFLite model to a file
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model TFLite has been saved with int8 quantization at {tflite_model_path}")
