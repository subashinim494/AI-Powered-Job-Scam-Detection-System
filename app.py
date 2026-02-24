
import os
import pickle
import tensorflow as tf
import requests
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# -------------------- BASIC SETUP --------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
app = Flask(__name__)

# -------------------- LOAD TOKENIZER --------------------
print("Loading tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded!")

# -------------------- LOAD MODEL --------------------
print("Loading model...")
model = tf.keras.models.load_model("fake_job_lstm_model.h5")
print("Model loaded!")

MAX_SEQUENCE_LENGTH = 200

# -------------------- PREPROCESS TEXT --------------------
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

# -------------------- FAKE LINK DETECTION --------------------
FAKE_KEYWORDS = [
    "earn", "urgent", "instant", "registration", "apply-now",
    "whatsapp", "telegram", "fee", "payment",
    "work-from-home", "no-interview", "hiring-fast"
]

SUSPICIOUS_TLDS = [
    ".xyz", ".site", ".online", ".info", ".top", ".buzz", ".work", ".today"
]

TRUSTED_DOMAINS = [
    "linkedin.com",
    "indeed.com",
    "naukri.com",
    "monster.com",
    "glassdoor.com",
    "careers.google.com",
    "amazon.jobs",
    "microsoft.com"
]

def is_fake_link(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        full_url = url.lower()

        # Trusted domains → SAFE
        for trusted in TRUSTED_DOMAINS:
            if trusted in domain:
                return False

        # Suspicious TLD
        for tld in SUSPICIOUS_TLDS:
            if domain.endswith(tld):
                return True

        # Suspicious keywords
        for word in FAKE_KEYWORDS:
            if word in full_url:
                return True

        # Too many hyphens
        if domain.count("-") >= 2:
            return True

        # Looks like a form-only site
        if "form" in full_url or "apply" in full_url:
            return True

        return False

    except Exception:
        return True

# -------------------- EXTRACT TEXT FROM LINK --------------------
def extract_text_from_link(url):
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        response = requests.get(url, headers=headers, timeout=15, verify=False)

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.content, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        cleaned_text = " ".join(text.split())

        return cleaned_text

    except Exception as e:
        print("Extraction error:", e)
        return ""

# -------------------- ROUTES --------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    job_link = request.form.get("job_link", "").strip()
    job_text = request.form.get("combined_text", "").strip()

    # -------- CASE 1: LINK GIVEN --------
    if job_link:
        if is_fake_link(job_link):
            return render_template(
                "index.html",
                prediction="The job post is FRAUDULENT"
            )

        extracted_text = extract_text_from_link(job_link)

        if not extracted_text:
            return render_template(
                "index.html",
                prediction="Unable to extract content from the link. Please try job description."
            )

        text_to_analyze = extracted_text

    # -------- CASE 2: DESCRIPTION GIVEN --------
    elif job_text:
        text_to_analyze = job_text

    # -------- CASE 3: NOTHING GIVEN --------
    else:
        return render_template(
            "index.html",
            prediction="Please enter either a job link or job description."
        )

    # -------- ML PREDICTION --------
    input_data = preprocess_text(text_to_analyze)
    score = model.predict(input_data)[0][0]

    result = "Fraudulent" if score > 0.7 else "Legitimate"

    return render_template(
        "index.html",
        prediction=f"The job post is {result}"
    )

# -------------------- RUN APP --------------------
if __name__ == "__main__":
    if __name__ == "__main__":
      app.run(host="0.0.0.0", port=5000)
