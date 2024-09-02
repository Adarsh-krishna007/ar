from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from PIL import Image
import io

app = Flask(__name__)

# Load TensorFlow model and label map
TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
IMAGE_SHAPE = (321, 321)

df = pd.read_csv(LABEL_MAP_URL)
label_map = dict(zip(df.id, df.name))

classifer = tf.keras.Sequential([
    hub.KerasLayer(TF_MODEL_URL, input_shape=IMAGE_SHAPE + (3,), output_key="predictions:logits")
])

def classify_img(img):
    img = np.array(img) / 255
    img = np.reshape(img, (1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))
    prediction = classifer.predict(img)
    return label_map[np.argmax(prediction)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    img = Image.open(file.stream)
    img = img.resize(IMAGE_SHAPE)
    result = classify_img(img)

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
