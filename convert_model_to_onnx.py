"""One-time development export of the Keras inference model to ONNX.

TensorFlow and tf2onnx are development-only dependencies. The deployed Flask
app loads only acb_prediction_model.onnx with ONNX Runtime.
"""

from pathlib import Path

import tensorflow as tf
import tf2onnx


BASE_DIR = Path(__file__).resolve().parent
KERAS_MODEL = BASE_DIR / "acb_prediction_model.h5"
ONNX_MODEL = BASE_DIR / "acb_prediction_model.onnx"
INPUT_SIGNATURE = (tf.TensorSpec((None, 160, 60, 1), tf.float32, name="image"),)


def main() -> None:
    model = tf.keras.models.load_model(KERAS_MODEL, compile=False)
    tf2onnx.convert.from_keras(
        model,
        input_signature=INPUT_SIGNATURE,
        opset=13,
        output_path=str(ONNX_MODEL),
    )
    print(f"Wrote {ONNX_MODEL}")


if __name__ == "__main__":
    main()
