# ACB CAPTCHA Solver

`index.py` is the production Flask entrypoint for Vercel. It exposes `app` and
uses `acb_prediction_model.onnx` with ONNX Runtime only.

`app.py` is the legacy TensorFlow/Keras reference implementation. The
development-only `convert_model_to_onnx.py` and `verify_onnx_model.py` scripts
respectively export and compare the inference model; TensorFlow is deliberately
absent from `requirements.txt` and excluded from the Vercel function bundle.
The legacy `.h5` models are kept locally for those development scripts but are
not tracked or deployed.
