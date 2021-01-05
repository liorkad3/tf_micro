import tensorflow as tf
import os
import cv2
import numpy as np

from src.slim_320 import create_slim_net


# Define paths to model files
MODELS_DIR = 'models/'
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
MODEL_TF = MODELS_DIR + 'model'
MODEL_NO_QUANT_TFLITE = MODELS_DIR + 'model_no_quant.tflite'
MODEL_TFLITE = MODELS_DIR + 'model.tflite'
MODEL_TFLITE_MICRO = MODELS_DIR + 'model.cc'

# model = tf.keras.models.load_model(MODEL_TF)


input_shape = (240, 320)  # H,W
base_channel = 8 * 2
num_classes = 2

model = create_slim_net(input_shape, base_channel, num_classes)

# tf.random.set_seed(5)
# inp = tf.random.normal([1, 240, 320, 3], 0, 1, tf.float32)

img = cv2.imread('test.jpg')
h, w, _ = img.shape
img_resize = cv2.resize(img, (320, 240))
img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
img_resize = img_resize - 127.0
img_resize = img_resize / 128.0

results = model.predict(np.expand_dims(img_resize, axis=0))  # result=[background,face,x1,y1,x2,y2]
print(results[0].shape, results[1].shape)

# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# model_no_quant_tflite = converter.convert()

# # Save the model to disk
# open(MODEL_NO_QUANT_TFLITE, "wb").write(model_no_quant_tflite)

# Convert the model to the TensorFlow Lite format with quantization
def representative_dataset():
    img = cv2.imread('test.jpg')
    h, w, _ = img.shape
    img_resize = cv2.resize(img, (320, 240))
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    img_resize = img_resize - 127.0
    img_resize = img_resize / 128.0
    img_resize = img_resize.astype(np.float32)
    for i in range(160):
        yield [np.expand_dims(img_resize, axis=0)]
        # yield([x_train[i].reshape(1, 1)])

# Set the optimization flag.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Enforce integer only quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
# Provide a representative dataset to ensure we quantize correctly.
converter.representative_dataset = representative_dataset
model_tflite = converter.convert()

# Save the model to disk
open(MODEL_TFLITE, "wb").write(model_tflite)