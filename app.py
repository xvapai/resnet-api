from flask import Flask, request, jsonify
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = ResNet50(weights='imagenet')

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = request.files['image']
    img_path = os.path.join("temp.jpg")
    img.save(img_path)

    img_loaded = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img_loaded)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]
    results = [{"label": label, "description": desc, "confidence": float(conf)} for (label, desc, conf) in decoded]
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
