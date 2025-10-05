
# app.py - Streamlit chatbot with runtime-safe model loading

import os
import sys
import subprocess
import importlib
import logging
import random
import json
import pickle
import numpy as np
import streamlit as st
from keras.models import load_model

# ------------------ ENV / LOGGING ------------------

# Suppress noisy TF logs before importing keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Utility: ensure package is installed
def ensure_package(name, version=None):
    try:
        return importlib.import_module(name)
    except Exception:
        pkg_spec = f"{name}=={version}" if version else name
        print(f"[startup] Package {name} not found — installing {pkg_spec} ...", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_spec])
        return importlib.import_module(name)

# ------------------ NLTK SETUP ------------------

nltk = None
try:
    nltk = importlib.import_module("nltk")
except Exception:
    nltk = ensure_package("nltk", "3.8.1")

from nltk.stem import WordNetLemmatizer

base_dir = os.path.dirname(__file__) or "."
nltk_data_path = os.path.join(base_dir, "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# Ensure corpora exist
_nltk_targets = {
    "punkt": "tokenizers/punkt",
    "wordnet": "corpora/wordnet",
    "stopwords": "corpora/stopwords",
    "omw-1.4": "corpora/omw-1.4",
}
for pkg, find_path in _nltk_targets.items():
    try:
        nltk.data.find(find_path)
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_path)

lemmatizer = WordNetLemmatizer()
ERROR_THRESHOLD = 0.25

# ------------------ STREAMLIT APP ------------------

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot")
st.write("Talk with the trained chatbot below!")
st.caption(f"Server Python: {sys.version.splitlines()[0]}")

# ------------------ SAFE LOAD HELPERS ------------------

def safe_load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load `{path}`: {e}")
        return None

def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load `{path}`: {e}")
        return None

words = safe_load_pickle(os.path.join(base_dir, "words.pkl"))
classes = safe_load_pickle(os.path.join(base_dir, "classes.pkl"))
intents = safe_load_json(os.path.join(base_dir, "intents.json"))

# ------------------ MODEL LOADING ------------------

model = None
try:
    from keras.models import load_model
    model_path_keras = os.path.join(base_dir, "General_chatbot.keras")
    model_path_h5 = os.path.join(base_dir, "General_chatbot.h5")

    if os.path.exists(model_path_keras):
        model = load_model(model_path_keras)   # Preferred modern format
        st.success("✅ Loaded model: General_chatbot.keras")
    elif os.path.exists(model_path_h5):
        # legacy fallback with compile=False to avoid batch_shape error
        model = load_model(model_path_h5, compile=False)
        st.warning("⚠️ Loaded legacy model: General_chatbot.h5 (re-save as .keras recommended)")
    else:
        st.error("❌ No model file found. Please upload `General_chatbot.keras` or `General_chatbot.h5`.")
except Exception as e:
    st.error(f"❌ Failed to load Keras model: {e}")
    model = None

# ------------------ NLP / BOT LOGIC ------------------

def clean_up_sentence(sentence):
    if not sentence:
        return []
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence):
    if words is None:
        return np.array([])
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    if model is None or words is None or classes is None:
        return []
    bow = bag_of_words(sentence)
    if bow.size == 0:
        return []
    res = model.predict(np.array([bow]), verbose=0)[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(ints, intents_json):
    if not ints:
        return "I didn’t understand that. Can you rephrase?"
    tag = ints[0]['intent']
    if not intents_json:
        return "Bot resources are missing (intents.json)."
    for i in intents_json.get('intents', []):
        if i.get('tag') == tag:
            return random.choice(i.get('responses', ["I didn't understand that."]))
    return "I didn’t understand that. Can you rephrase?"

# ------------------ STREAMLIT CHAT UI ------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    ints = predict_class(user_input)
    res = get_response(ints, intents)
    st.session_state["messages"].append({"role": "bot", "content": res})

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

