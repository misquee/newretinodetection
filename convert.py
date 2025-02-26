import tensorflowjs as tfjs
import tensorflow as tf

# Load model H5
model = tf.keras.models.load_model('model_percobaanx.h5')

# Konversi ke format TensorFlow.js
tfjs.converters.save_keras_model(model, 'model_tfjs')
