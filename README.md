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

To reuse an already verified device session instead of logging in, pass
`cookie` (the complete browser Cookie header), `dse_session_id`, and
`account_number`. For a private deployment, set `ACB_COOKIE_HEADER` and
`ACB_DSE_SESSION_ID` as deployment secrets instead. The API tries this session
directly; it does not trigger the CAPTCHA/login flow unless that session expires.
If you also supply credentials, it logs in once only after the trusted session
expires. Never commit cookies/tokens, and do not expose this API publicly.

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
