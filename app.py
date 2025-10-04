import random
import json
import pickle
import numpy as np
import nltk
import streamlit as st
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Initialize
lemmatizer = WordNetLemmatizer()
ERROR_THRESHOLD = 0.25

# Load data
with open('intents.json') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('General_chatbot.h5')

# Preprocessing
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Prediction
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Response
def get_response(ints, intents_json):
    if len(ints) == 0:
        return "I didn’t understand that. Can you rephrase?"

    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I didn’t understand that. Can you rephrase?"


# ------------------ STREAMLIT UI ------------------

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot")
st.write("Talk with the trained chatbot below!")

# Store conversation in session
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# User input box at bottom
user_input = st.chat_input("Type your message...")

if user_input:
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Bot response
    ints = predict_class(user_input)
    res = get_response(ints, intents)
    st.session_state["messages"].append({"role": "bot", "content": res})

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])
