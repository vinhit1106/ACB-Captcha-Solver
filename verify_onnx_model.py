"""Development-only parity check between the Keras and ONNX inference models."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorflow as tf

from index import decode_predictions


BASE_DIR = Path(__file__).resolve().parent


def keras_decode(predictions: np.ndarray) -> list[str]:
    characters = [
        " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "a", "b", "d", "e", "f", "g", "h", "k", "l", "m", "n", "o",
        "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    ]
    vocabulary = ["[UNK]", *characters]
    decoded = tf.keras.backend.ctc_decode(
        predictions,
        input_length=np.full(predictions.shape[0], predictions.shape[1]),
        greedy=True,
    )[0][0].numpy()[:, :6]
    return ["".join(vocabulary[index] for index in row if index >= 0) for row in decoded]


def main() -> None:
    keras_model = tf.keras.models.load_model(BASE_DIR / "acb_prediction_model.h5", compile=False)
    session = ort.InferenceSession(
        str(BASE_DIR / "acb_prediction_model.onnx"), providers=["CPUExecutionProvider"]
    )
    batch = np.random.default_rng(0).random((2, 160, 60, 1), dtype=np.float32)
    keras_output = keras_model(batch, training=False).numpy()
    onnx_output = session.run(None, {session.get_inputs()[0].name: batch})[0]

    np.testing.assert_allclose(onnx_output, keras_output, rtol=1e-4, atol=1e-5)
    assert decode_predictions(onnx_output) == keras_decode(keras_output)
    print("ONNX output and greedy CTC decoding match Keras.")


if __name__ == "__main__":
    main()
