import os
import pickle
import tensorflow as tf
import gradio as gr
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("Loading tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded!")

print("Loading model...")
model = tf.keras.models.load_model("fake_job_lstm_model.h5")
print("Model loaded!")

MAX_SEQUENCE_LENGTH = 200

def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

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

        for trusted in TRUSTED_DOMAINS:
            if trusted in domain:
                return False

        for tld in SUSPICIOUS_TLDS:
            if domain.endswith(tld):
                return True

        for word in FAKE_KEYWORDS:
            if word in full_url:
                return True

        if domain.count("-") >= 2:
            return True

        if "form" in full_url or "apply" in full_url:
            return True

        return False
    except:
        return True

def extract_text_from_link(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.content, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        cleaned_text = " ".join(text.split())
        return cleaned_text

    except:
        return ""

def predict(job_link, job_text):

    if job_link:
        if is_fake_link(job_link):
            return "The job post is FRAUDULENT ❌"

        extracted_text = extract_text_from_link(job_link)

        if not extracted_text:
            return "Unable to extract content from the link."

        text_to_analyze = extracted_text

    elif job_text:
        text_to_analyze = job_text

    else:
        return "Please enter either a job link or job description."

    input_data = preprocess_text(text_to_analyze)
    score = model.predict(input_data)[0][0]

    result = "Fraudulent ❌" if score > 0.7 else "Legitimate ✅"
    return f"The job post is {result}"

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Job Link (Optional)"),
        gr.Textbox(label="Job Description (Optional)", lines=5)
    ],
    outputs="text",
    title="AI Powered Job Scam Detection System"
)

interface.launch()
