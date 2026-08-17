import base64
import os
import time
from io import BytesIO

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "acb_prediction_model.onnx")
IMG_WIDTH = 160
IMG_HEIGHT = 60
MAX_LENGTH = 6

# This is the vocabulary returned by Keras StringLookup during training. Index
# zero is StringLookup's unknown token; the final model class is the CTC blank.
CHARACTERS = [
    " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "d", "e", "f", "g", "h", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
]
INDEX_TO_CHARACTER = ["[UNK]", *CHARACTERS]


def load_session() -> ort.InferenceSession:
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(
            "ONNX model not found. Run convert_model_to_onnx.py during development "
            "and commit acb_prediction_model.onnx."
        )
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_metadata = session.get_inputs()[0]
    output_metadata = session.get_outputs()[0]
    if input_metadata.shape[-3:] != [IMG_WIDTH, IMG_HEIGHT, 1]:
        raise RuntimeError(f"Unexpected ONNX input shape: {input_metadata.shape}")
    if output_metadata.shape[-1] != len(INDEX_TO_CHARACTER) + 1:
        raise RuntimeError(f"Unexpected ONNX output shape: {output_metadata.shape}")
    return session


SESSION = load_session()
INPUT_NAME = SESSION.get_inputs()[0].name


def preprocess_image(base64_image: str) -> np.ndarray:
    try:
        if "," in base64_image:
            base64_image = base64_image.split(",", 1)[1]
        encoded = base64_image.replace("-", "+").replace("_", "/")
        encoded += "=" * (-len(encoded) % 4)
        image = Image.open(BytesIO(base64.b64decode(encoded, validate=True))).convert("L")
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        return np.expand_dims(pixels.T, axis=-1)
    except Exception as exc:
        raise ValueError(f"Error processing image: {exc}") from exc


def decode_predictions(predictions: np.ndarray) -> list[str]:
    """Greedily decode Keras-compatible CTC output without TensorFlow."""
    blank_index = predictions.shape[-1] - 1
    decoded_texts = []
    for sequence in np.argmax(predictions, axis=-1):
        previous_index = blank_index
        characters = []
        for index in sequence:
            index = int(index)
            if index == blank_index:
                previous_index = index
                continue
            if index < 0 or index >= len(INDEX_TO_CHARACTER):
                previous_index = index
                continue
            if index != previous_index:
                characters.append(INDEX_TO_CHARACTER[index])
                if len(characters) == MAX_LENGTH:
                    break
            previous_index = index
        decoded_texts.append("".join(characters))
    return decoded_texts


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify(status="success", message="ACB CAPTCHA Solver API is running")


@app.route("/api/captcha/solve", methods=["POST"])
def solve_captcha():
    started_at = time.time()
    try:
        content = request.get_json(silent=True) or {}
        if "base64" not in content:
            return jsonify(status="error", message="Missing 'base64' field"), 400
        try:
            image = preprocess_image(content["base64"])
        except ValueError as exc:
            return jsonify(status="error", message=str(exc)), 400
        batch = np.expand_dims(image, axis=0).astype(np.float32, copy=False)
        predictions = SESSION.run(None, {INPUT_NAME: batch})[0]
        captcha = decode_predictions(predictions)[0].replace("[UNK]", "").replace(" ", "")
        return jsonify(
            status="success",
            captcha=captcha,
            latency_ms=int((time.time() - started_at) * 1000),
        )
    except Exception as exc:
        return jsonify(status="error", message=str(exc)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8888")))
