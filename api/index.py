import os
import sys
import time
import base64
from io import BytesIO

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

layers = keras.layers
register_keras_serializable = keras.utils.register_keras_serializable

LSTM = layers.LSTM
Bidirectional = layers.Bidirectional
Dense = layers.Dense


# =========================
# PATH
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "acb_model.h5"
)


# =========================
# CONFIG
# =========================

IMG_WIDTH = 160
IMG_HEIGHT = 60
MAX_LENGTH = 6

CHARACTERS = [
    " ",
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","D","E","F","G","H","I","J","K",
    "L","M","N","O","P","Q","R","S","T","U",
    "V","W","X","Y","Z",
    "a","b","d","e","f","g","h","k","l","m",
    "n","o","p","q","r","s","t","u","v","w",
    "x","y","z"
]


# =========================
# STRING LOOKUP
# =========================

char_to_num = layers.StringLookup(
    vocabulary=CHARACTERS,
    mask_token=None
)

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True
)


# =========================
# CUSTOM CTC
# =========================

@register_keras_serializable(package="custom")
class CTCLayer(layers.Layer):

    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):

        batch_len = tf.cast(
            tf.shape(y_true)[0],
            dtype="int64"
        )

        input_length = tf.cast(
            tf.shape(y_pred)[1],
            dtype="int64"
        )

        label_length = tf.cast(
            tf.shape(y_true)[1],
            dtype="int64"
        )

        input_length = input_length * tf.ones(
            shape=(batch_len, 1),
            dtype="int64"
        )

        label_length = label_length * tf.ones(
            shape=(batch_len, 1),
            dtype="int64"
        )

        loss = self.loss_fn(
            y_true,
            y_pred,
            input_length,
            label_length
        )

        self.add_loss(loss)

        return y_pred


# =========================
# LOAD MODEL
# =========================

custom_objs = {
    "CTCLayer": CTCLayer,
    "LSTM": LSTM,
    "Bidirectional": Bidirectional,
    "Dense": Dense,
}

print("Loading TensorFlow model...")

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects=custom_objs,
    compile=False
)

prediction_model = keras.models.Model(
    model.get_layer(name="image").input,
    model.get_layer(name="dense2").output
)

print("Model loaded successfully")


# =========================
# IMAGE PREPROCESS
# =========================

def preprocess_image(base64_image: str):

    try:

        # nếu frontend gửi:
        # data:image/png;base64,xxxx
        if "," in base64_image:
            base64_image = base64_image.split(",", 1)[1]

        b64 = base64_image.replace("-", "+").replace("_", "/")

        # fix thiếu padding
        b64 += "=" * (-len(b64) % 4)

        img_data = base64.b64decode(b64)

        img = Image.open(
            BytesIO(img_data)
        ).convert("L")

        img = img.resize(
            (IMG_WIDTH, IMG_HEIGHT)
        )

        arr = np.array(
            img,
            dtype=np.float32
        ) / 255.0

        arr = np.expand_dims(
            arr.T,
            axis=-1
        )

        return arr

    except Exception as e:

        raise ValueError(
            f"Error processing image: {str(e)}"
        )


# =========================
# DECODE
# =========================

def decode_predictions(pred):

    input_len = np.ones(
        pred.shape[0]
    ) * pred.shape[1]

    decoded = keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=True
    )[0][0][:, :MAX_LENGTH]

    outputs = []

    for res in decoded:

        text = tf.strings.reduce_join(
            num_to_char(res)
        ).numpy().decode("utf-8")

        outputs.append(text)

    return outputs


# =========================
# FLASK APP
# =========================

app = Flask(__name__)

CORS(app)


@app.route("/", methods=["GET"])
def home():

    return jsonify(
        status="success",
        message="ACB CAPTCHA Solver API is running"
    )


@app.route("/api/captcha/solve", methods=["POST"])
def solve_captcha():

    t0 = time.time()

    try:

        content = request.get_json(
            silent=True
        ) or {}

        if "base64" not in content:

            return jsonify(
                status="error",
                message="Missing 'base64' field"
            ), 400


        img = preprocess_image(
            content["base64"]
        )

        batch = np.array([
            img
        ])


        preds = prediction_model.predict(
            batch,
            verbose=0
        )

        texts = decode_predictions(
            preds
        )

        captcha = texts[0] \
            .replace("[UNK]", "") \
            .replace(" ", "")


        return jsonify(
            status="success",
            captcha=captcha,
            latency_ms=int(
                (time.time() - t0) * 1000
            )
        )


    except Exception as e:

        return jsonify(
            status="error",
            message=str(e)
        ), 500