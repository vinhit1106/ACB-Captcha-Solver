# CAPTCHA SERVER SOLVER FOR IMAGE CAPTCHAS
# - Portable model path for PyInstaller
# - Uses tf.keras only (no standalone keras), with a registered custom CTC layer
# - Flask API: POST /api/captcha/solve { "base64": "<base64 data>" }

import os
import sys
import time
import base64
from io import BytesIO

# --- Quiet TensorFlow logs (set BEFORE importing TF) ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # 0=all,1=INFO,2=WARNING,3=ERROR
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # silence oneDNN differences note

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

# Use tf.keras consistently
layers = keras.layers
register_keras_serializable = keras.utils.register_keras_serializable
LSTM = layers.LSTM
Bidirectional = layers.Bidirectional
Dense = layers.Dense

# -----------------------
# Utilities / Configuration
# -----------------------
def resource_path(rel: str) -> str:
    """Return absolute path to resource for dev and PyInstaller (_MEIPASS) runtime."""
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel)

# -----------------------
# Constants
# -----------------------
BATCH_SIZE = 16
IMG_WIDTH = 160
IMG_HEIGHT = 60
DOWNSAMPLE_FACTOR = 4
MAX_LENGTH = 6  # Adjust based on your CAPTCHA length

# Place acb_model.h5 next to the exe (or script) and bundle via --add-data "acb_model.h5;."
MODEL_PATH = resource_path("acb_model.h5")

# Character set (leading space kept so StringLookup vocab matches your training)
CHARACTERS = [
    " ", "0","1","2","3","4","5","6","7","8","9",
    "A","B","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "a","b","d","e","f","g","h","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"
]

# -----------------------
# Keras layers & mappings
# -----------------------
char_to_num = layers.StringLookup(vocabulary=CHARACTERS, mask_token=None)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

@register_keras_serializable(package="custom")
class CTCLayer(layers.Layer):
    """CTC loss layer compatible with tf.keras saving/loading."""
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

# -----------------------
# Model Loading
# -----------------------
# compile=False avoids needing original optimizer/loss when only doing inference.
custom_objs = {
    "CTCLayer": CTCLayer,
    # include common built-ins your model uses so legacy .h5 deserializer never complains
    "LSTM": LSTM,
    "Bidirectional": Bidirectional,
    "Dense": Dense,
}
model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objs, compile=False)

# Use the inference head (adjust layer names if your model differs)
prediction_model = keras.models.Model(
    model.get_layer(name="image").input,
    model.get_layer(name="dense2").output
)

# -----------------------
# Inference helpers
# -----------------------
def preprocess_image(base64_image: str) -> np.ndarray:
    """
    Decode a base64 image to model-ready array:
    - URL-safe to standard base64
    - convert to grayscale
    - resize to (IMG_WIDTH, IMG_HEIGHT)
    - normalize to [0,1]
    - transpose (W,H)->(H,W) if your CRNN expects time along width
    - add channel dimension -> (W, H, 1)
    """
    try:
        b64 = base64_image.replace("-", "+").replace("_", "/")
        img_data = base64.b64decode(b64)
        img = Image.open(BytesIO(img_data)).convert("L")
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr.T, axis=-1)  # shape: (width, height, 1)
        return arr
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def decode_predictions(pred: np.ndarray) -> list[str]:
    """Greedy CTC decode to string; trims to MAX_LENGTH."""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded = keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True
    )[0][0][:, :MAX_LENGTH]

    outputs = []
    for res in decoded:
        text = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        outputs.append(text)
    return outputs

# -----------------------
# Flask App
# -----------------------
app = Flask(__name__)
CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

@app.route("/api/captcha/solve", methods=["POST"])
@cross_origin(origin="*")
def solve_captcha():
    t0 = time.time()
    try:
        content = request.get_json(silent=True) or {}
        if "base64" not in content:
            return jsonify(status="error", message="Missing 'base64' field"), 400

        img = preprocess_image(content["base64"])
        batch = np.array([img])  # (1, W, H, 1)

        preds = prediction_model.predict(batch, verbose=0)
        texts = decode_predictions(preds)

        captcha = texts[0].replace("[UNK]", "").replace(" ", "")

        return jsonify(
            status="success",
            captcha=captcha,
            latency_ms=int((time.time() - t0) * 1000)
        )
    except Exception as e:
        return jsonify(status="error", message=str(e)), 500

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    print("Starting server on http://localhost:8888")
    print("Press Ctrl+C to quit")
    app.run(host="0.0.0.0", port=8888)