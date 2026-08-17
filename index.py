"""Unofficial ACB transaction API with internal ONNX CAPTCHA solving."""

import base64
import os
import time
from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import numpy as np
import onnxruntime as ort
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "acb_prediction_model.onnx")
IMG_WIDTH, IMG_HEIGHT, MAX_LENGTH = 160, 60, 6
BASE_URL = "https://online.acb.com.vn"
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
DEFAULT_CAPTCHA_RETRIES = int(os.getenv("MAX_CAPTCHA_RETRIES", "2"))

# Exact Keras StringLookup vocabulary. The final ONNX output class is CTC blank.
CHARACTERS = [
    " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "d", "e", "f", "g", "h", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
]
INDEX_TO_CHARACTER = ["[UNK]", *CHARACTERS]


class ACBError(Exception):
    def __init__(self, code: str, message: str, status: int = 502):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def load_session() -> ort.InferenceSession:
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError("ONNX model is missing from the application bundle.")
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_meta, output_meta = session.get_inputs()[0], session.get_outputs()[0]
    if input_meta.shape[-3:] != [IMG_WIDTH, IMG_HEIGHT, 1]:
        raise RuntimeError(f"Unexpected ONNX input shape: {input_meta.shape}")
    if output_meta.shape[-1] != len(INDEX_TO_CHARACTER) + 1:
        raise RuntimeError(f"Unexpected ONNX output shape: {output_meta.shape}")
    return session


SESSION = load_session()
INPUT_NAME = SESSION.get_inputs()[0].name


def preprocess_image(base64_image: str) -> np.ndarray:
    """Exact Pillow preprocessing used by the verified ONNX solver."""
    try:
        if "," in base64_image:
            base64_image = base64_image.split(",", 1)[1]
        encoded = base64_image.replace("-", "+").replace("_", "/")
        encoded += "=" * (-len(encoded) % 4)
        image = Image.open(BytesIO(base64.b64decode(encoded, validate=True))).convert("L")
        pixels = np.asarray(image.resize((IMG_WIDTH, IMG_HEIGHT)), dtype=np.float32) / 255.0
        return np.expand_dims(pixels.T, axis=-1)
    except Exception as exc:
        raise ValueError(f"Invalid CAPTCHA image: {exc}") from exc


def decode_predictions(predictions: np.ndarray) -> list[str]:
    """Greedily decode Keras-compatible CTC output without TensorFlow."""
    blank_index, decoded_texts = predictions.shape[-1] - 1, []
    for sequence in np.argmax(predictions, axis=-1):
        previous_index, characters = blank_index, []
        for raw_index in sequence:
            index = int(raw_index)
            if index == blank_index:
                previous_index = index
                continue
            if 0 <= index < len(INDEX_TO_CHARACTER) and index != previous_index:
                characters.append(INDEX_TO_CHARACTER[index])
                if len(characters) == MAX_LENGTH:
                    break
            previous_index = index
        decoded_texts.append("".join(characters))
    return decoded_texts


def solve_captcha_bytes(image: bytes) -> str:
    tensor = preprocess_image(base64.b64encode(image).decode("ascii"))
    batch = np.expand_dims(tensor, axis=0).astype(np.float32, copy=False)
    captcha = decode_predictions(SESSION.run(None, {INPUT_NAME: batch})[0])[0]
    captcha = captcha.replace("[UNK]", "").replace(" ", "").upper()
    if not captcha:
        raise ACBError("captcha_decode_failed", "Could not decode the CAPTCHA image.")
    return captcha


class ACBClient:
    """One isolated ACB session; never persists or exposes cookies/tokens."""

    def __init__(self, username: str, password: str, max_captcha_retries: int):
        self.username, self.password = username, password
        self.max_captcha_retries = max_captcha_retries
        self.session_id: str | None = None
        self.login_page_url: str | None = None
        self.http = requests.Session()
        retry = Retry(total=2, connect=2, read=2, backoff_factor=0.3,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset({"GET"}))
        self.http.mount("https://", HTTPAdapter(max_retries=retry))
        self.http.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
            "Accept-Language": "en-GB,en;q=0.9,en-US;q=0.8,vi;q=0.7",
        })

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        try:
            response = self.http.request(method, url, timeout=DEFAULT_TIMEOUT_SECONDS,
                                         allow_redirects=False, **kwargs)
        except requests.RequestException as exc:
            raise ACBError("acb_unreachable", "Unable to reach ACB service.") from exc
        if response.status_code >= 500:
            raise ACBError("acb_unavailable", "ACB service is temporarily unavailable.")
        return response

    @staticmethod
    def _normalize_url(location: str) -> str:
        if location.startswith("http://online.acb.com.vn:443"):
            return location.replace("http://online.acb.com.vn:443", BASE_URL, 1)
        if location.startswith("http://"):
            return "https://" + location[len("http://"):]
        return urljoin(BASE_URL, location)

    def init_session(self) -> None:
        response = self._request("GET", f"{BASE_URL}/acbib/Request")
        if response.status_code not in range(200, 400):
            raise ACBError("session_initialization_failed", "Could not initialize ACB session.")

    def get_login_page(self) -> None:
        response = self._request("GET", f"{BASE_URL}/acbib/webmbtt")
        if response.status_code == 302 and response.headers.get("Location"):
            self.login_page_url = self._normalize_url(response.headers["Location"])
            response = self._request("GET", self.login_page_url,
                                     headers={"Referer": f"{BASE_URL}/acbib/webmbtt"})
        else:
            self.login_page_url = response.url
        self.session_id = parse_qs(urlparse(self.login_page_url).query).get("dse_sessionId", [None])[0]
        if not self.session_id:
            raise ACBError("session_id_missing", "ACB did not return a login session.")
        if response.status_code != 200:
            raise ACBError("login_page_failed", "Could not load ACB login page.")

    def download_captcha(self) -> bytes:
        response = self._request("GET", f"{BASE_URL}/acbib/Captcha.jpg", headers={
            "Referer": self.login_page_url or BASE_URL,
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        })
        if response.status_code != 200 or not response.content:
            raise ACBError("captcha_missing", "ACB did not return a CAPTCHA image.")
        return response.content

    @staticmethod
    def _check_login_response(html: str) -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        labels = " ".join(label.get_text() for label in soup.select("label[for='safekey']"))
        if soup.select_one("#safekey") or "OTPSafekey" in labels:
            return {"requires_otp": True}
        holder = soup.select_one(".content-holder")
        if holder:
            return {"success": True}
        body = soup.get_text("", strip=True)
        if "Mãxácthựckhôngđúng" in body or "Securitycode" in body:
            return {"wrong_captcha": True}
        error = soup.select_one(".error-message,.alert-danger,.error,.loginError")
        return {"message": error.get_text(" ", strip=True) if error else "ACB rejected the login."}

    def submit_login(self, captcha: str) -> dict[str, Any]:
        payload = {
            "dse_sessionId": self.session_id, "dse_applicationId": "-1", "dse_pageId": "2",
            "dse_operationName": "obkLoginOp", "dse_errorPage": "ibk/login.jsp",
            "dse_processorState": "initial", "UserName": self.username, "PassWord": self.password,
            "glbLogedIn": "WEB", "SecurityCode": captcha,
        }
        response = self._request("POST", f"{BASE_URL}/acbib/Request", data=payload, headers={
            "Origin": BASE_URL, "Referer": self.login_page_url or BASE_URL,
            "Content-Type": "application/x-www-form-urlencoded",
        })
        if response.status_code == 302 and response.headers.get("Location"):
            response = self._request("GET", self._normalize_url(response.headers["Location"]),
                                     headers={"Referer": f"{BASE_URL}/acbib/Request"})
        return self._check_login_response(response.text)

    def login(self) -> None:
        self.init_session()
        self.get_login_page()
        for _ in range(self.max_captcha_retries):
            result = self.submit_login(solve_captcha_bytes(self.download_captcha()))
            if result.get("success"):
                return
            if result.get("requires_otp"):
                raise ACBError("otp_required", "ACB requires OTP verification for this login.", 401)
            if not result.get("wrong_captcha"):
                raise ACBError("login_failed", result.get("message", "ACB rejected the login."), 401)
        raise ACBError("captcha_rejected", "ACB rejected the CAPTCHA after the configured retry limit.", 401)

    def get_transactions(self, from_date: date, to_date: date, account_number: str) -> list[dict[str, Any]]:
        params = {
            "dse_sessionId": self.session_id, "dse_applicationId": "-1", "dse_pageId": "2",
            "dse_operationName": "ibkacctDetailProc", "dse_errorPage": "login.jsp",
            "dse_processorState": "initial", "dse_nextEventName": "start", "AccountNbr": account_number,
        }
        detail = self._request("GET", f"{BASE_URL}/acbib/Request", params=params)
        processor = BeautifulSoup(detail.text, "html.parser").select_one("input[name='dse_processorId']")
        processor_id = processor.get("value") if processor else None
        if not processor_id:
            raise ACBError("session_expired", "ACB session expired before transactions could be loaded.")
        payload = {
            "dse_sessionId": self.session_id, "dse_applicationId": "-1", "dse_operationName": "ibkacctDetailProc",
            "dse_pageId": "4", "dse_processorState": "acctDetailPage", "dse_processorId": processor_id,
            "dse_errorPage": "/ibk/acctinquiry/trans.jsp", "AccountNbr": account_number,
            "virtualAccount": "", "storeName": "", "CheckRef": "false", "EdtRef": "",
            "dse_nextEventName": "byDate", "activeDatetimeYN": "N",
            "FromDate": from_date.strftime("%d/%m/%Y"), "ToDate": to_date.strftime("%d/%m/%Y"),
        }
        response = self._request("POST", f"{BASE_URL}/acbib/Request", data=payload)
        return self.parse_transactions(response.text)

    @staticmethod
    def parse_transactions(html: str) -> list[dict[str, Any]]:
        table = BeautifulSoup(html, "html.parser").select_one("#table1")
        transactions: list[dict[str, Any]] = []
        if not table:
            return transactions
        def amount(value: str) -> int:
            try:
                return int(value.replace(" ", "").replace(".", "") or "0")
            except ValueError:
                return 0
        for row in table.select("tr.table-style-double1"):
            cells = row.find_all("td")
            if len(cells) == 6:
                values = [cell.get_text(" ", strip=True) for cell in cells]
                if values[2] and values[2] != "&nbsp;":
                    transactions.append({"effectiveDate": values[0], "transactionDate": values[1],
                        "transactionNumber": values[2], "debit": amount(values[3]),
                        "credit": amount(values[4]), "balance": amount(values[5]), "description": ""})
            elif transactions:
                description = row.select_one("td.acctSum") or (cells[0] if len(cells) == 2 else None)
                if description:
                    text = description.get_text(" ", strip=True)
                    if text and text != "&nbsp;":
                        transactions[-1]["description"] = text
        return transactions


def parse_iso_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ACBError("invalid_request", f"'{field}' must be YYYY-MM-DD.", 400)
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ACBError("invalid_request", f"'{field}' must be YYYY-MM-DD.", 400) from exc


app = Flask(__name__)


@app.get("/")
def home():
    return jsonify(status="success", message="Unofficial ACB transaction API is running")


@app.post("/api/transactions")
def transactions():
    started_at = time.time()
    try:
        content = request.get_json(silent=True) or {}
        username = content.get("username") or os.getenv("ACB_USERNAME")
        password = content.get("password") or os.getenv("ACB_PASSWORD")
        if not isinstance(username, str) or not isinstance(password, str) or not username or not password:
            raise ACBError("credentials_required", "Provide username and password, or configure server credentials.", 400)
        account_number = content.get("account_number") or username
        if not isinstance(account_number, str) or not account_number:
            raise ACBError("invalid_request", "'account_number' must be a non-empty string.", 400)
        to_date = parse_iso_date(content["to_date"], "to_date") if "to_date" in content else date.today()
        from_date = parse_iso_date(content["from_date"], "from_date") if "from_date" in content else to_date - timedelta(days=3)
        if from_date > to_date:
            raise ACBError("invalid_request", "'from_date' cannot be after 'to_date'.", 400)
        client = ACBClient(username, password, DEFAULT_CAPTCHA_RETRIES)
        client.login()
        try:
            result = client.get_transactions(from_date, to_date, account_number)
        except ACBError as exc:
            # ACB can invalidate a session between authenticated requests. Start
            # one clean login/session once; do not loop indefinitely.
            if exc.code != "session_expired":
                raise
            client = ACBClient(username, password, DEFAULT_CAPTCHA_RETRIES)
            client.login()
            result = client.get_transactions(from_date, to_date, account_number)
        return jsonify(status="success", account_number=account_number, from_date=from_date.isoformat(),
                       to_date=to_date.isoformat(), transactions=result,
                       latency_ms=int((time.time() - started_at) * 1000))
    except ACBError as exc:
        return jsonify(status="error", error=exc.code, message=exc.message), exc.status
    except Exception:
        app.logger.exception("Unhandled transaction request failure")
        return jsonify(status="error", error="internal_error", message="Unexpected server error."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8888")))
