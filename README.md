# Unofficial ACB Transaction API

One Flask API that retrieves ACB transaction history. CAPTCHA solving is entirely
internal: the caller does not run or call a second CAPTCHA service.

## Endpoint

`POST /api/transactions`

```json
{
  "username": "your-acb-username",
  "password": "your-acb-password",
  "account_number": "optional-account-number",
  "from_date": "2026-08-14",
  "to_date": "2026-08-17"
}
```

`account_number` defaults to `username`; dates default to the last three days.
The response contains `transactions` and never returns credentials, cookies, or
ACB tokens.

For a trusted private deployment, credentials can instead be set as
`ACB_USERNAME` and `ACB_PASSWORD` environment variables. Do not expose an
environment-configured deployment publicly without authentication in front of it.

## Internal flow

1. Create an isolated ACB HTTP session.
2. Load the login page and CAPTCHA.
3. Run `acb_prediction_model.onnx` through ONNX Runtime.
4. Submit the decoded CAPTCHA and complete login.
5. Fetch and parse transaction history.

Wrong CAPTCHA responses are retried up to `MAX_CAPTCHA_RETRIES` (default: 2).
If ACB expires the session during the transaction request, the API performs one
fresh login and retries that transaction request once. Cookies are intentionally
request-local, so they are not shared between callers or persisted to unreliable
serverless storage.

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python index.py
```

The server listens on port 8888 by default.

## Deployment

`index.py` is the Vercel Flask entrypoint. `vercel.json` includes the ONNX model
and excludes development-only TensorFlow artifacts. Production dependencies use
ONNX Runtime only; TensorFlow and Keras are not in `requirements.txt`.

The legacy TensorFlow conversion/parity scripts remain development references and
are not deployed.
