import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
import os
import json
import sys
import requests
from bs4 import BeautifulSoup
import trafilatura
from model import BertClassifier

MODEL_DIR = "saved_model"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "weights.h5")
CONFIG_PATH  = os.path.join(MODEL_DIR, "config.json")
TOKENIZER_DIR = os.path.join(MODEL_DIR, "tokenizer")

def extract_text_from_url(url):
    """Extracts main article text from a URL."""
    try:
        html = trafilatura.fetch_url(url)
        if html:
            text = trafilatura.extract(html, include_comments=False, include_tables=False, favor_precision=True)
            if text and len(text) > 100:
                return text
    except Exception:
        pass

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if len(text) > 100:
            return text
    except Exception as e:
        print("Extraction failed:", e)

    return None


def load_model_and_tokenizer():
    """Rebuilds the architecture and loads saved weights."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    num_labels = config["num_labels"]
    max_len = config["max_len"]
    label_names = config["label_names"]

    print("Building model architecture...")
    bert = TFBertModel.from_pretrained(config["base_model"])
    model = BertClassifier(bert, num_labels)

    dummy = {
        "input_ids": tf.zeros((1, max_len), dtype=tf.int32),
        "attention_mask": tf.ones((1, max_len), dtype=tf.int32),
    }
    _ = model(dummy)

    print("Loading trained weights...")
    model.load_weights(WEIGHTS_PATH)

    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)

    return model, tokenizer, max_len, label_names


def main(url):
    model, tokenizer, max_len, label_names = load_model_and_tokenizer()

    print("\nExtracting text from:", url)
    text = extract_text_from_url(url)
    if text is None:
        print("Could not extract text from this URL.")
        sys.exit(1)
    print("Extracted", len(text), "characters")

    inputs = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="tf",
    )

    prediction = model(inputs)
    probs = prediction.numpy()[0]
    predicted_index = int(np.argmax(probs))
    confidence = float(probs[predicted_index])
    predicted_label = label_names[predicted_index]

    label_to_id = {
    "اجتماعی": 0,
    "اقتصادی": 1,
    "بین‌الملل": 2,
    "سیاسی": 3,
    "علمی فناوری": 4,
    "فرهنگی هنری": 5,
    "ورزشی": 6,
    "پزشکی": 7,
    }
    print(label_to_id)

    print("\n==============================")
    print("Predicted category:", predicted_label)
    print("Confidence:", f"{confidence*100:.2f}%")
    print("------------------------------")
    print("All class probabilities:")
    for i, name in enumerate(label_names):
        bar = "#" * int(probs[i] * 30)
        print(f"  {name:<15} {probs[i]:.4f}  {bar}")
    print("==============================")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <URL>")
        sys.exit(1)
    main(sys.argv[1])

