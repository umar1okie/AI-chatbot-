import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras import layers, optimizers

# --- Initialize ---
lemmatizer = WordNetLemmatizer()

# Load intents.json
with open('intents.json') as file:
    intents = json.load(file)

words, classes, documents = [], [], []
ignoreLetters = ['?', '!', '.', ',']

# Tokenize and prepare data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize + sort
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Training data
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
    for word in words:
        bag.append(1 if word in wordPatterns else 0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# --- Build Model ---
model = tf.keras.Sequential([
    layers.Input(shape=(len(trainX[0]),)),   # 👈 replaces input_shape/batch_shape
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(trainY[0]), activation='softmax')
])
# Optimizer
sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Compile
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# --- Save Model (only .keras format) ---
model.save("General_chatbot.keras")
print("✅ Model trained & saved successfully in .keras format")
