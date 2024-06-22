from flask import Flask, request, jsonify
import string
import pickle
import numpy as np
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from flask_cors import CORS
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.stem import WordNetLemmatizer
import nltk


app = Flask(__name__)
CORS(app)

nltk_resources = ['punkt', 'wordnet', 'omw']

for resource in nltk_resources:
    if not nltk.download(resource, quiet=True):
        print(f"Downloaded {resource} corpus.")

print("All required NLTK resources are downloaded.")

# Muat model dan objek pendukung
model = load_model('chat_model_transformer.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
token = pickle.load(open('tokenizer.pkl', 'rb'))
responses = pickle.load(open('responses.pkl', 'rb'))
input_shape = model.input_shape[1]

# Load Indonesian stopwords
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# Lematisasi menggunakan WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# Fungsi pra-pemrosesan
def preprocess_text(text):
    # Menghapus tanda baca dan konversi ke huruf kecil
    text = ''.join([letters.lower() for letters in text if letters not in string.punctuation])
    # Menghapus stopword
    text = stopword_remover.remove(text)
    # Lematisasi
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text


@app.get('/')
def index_get():
    return jsonify({
        'API': "ini adalah api untuk chatbot universitas muhammadiyah sukabumi",
    })


@app.route('/predict', methods=['POST'])
def predict():
    prediction_input = request.get_json().get('message')
    # print(prediction_input+" ini input")
    if not input_shape:
        return jsonify({'error': 'No input text provided'}), 400
    texts_p = []
    # Menghapus punktuasi atau tanda baca dan konversi ke huruf kecil
    prediction_input = preprocess_text(prediction_input)
    # print(prediction_input)
    texts_p.append(prediction_input)
    # print(texts_p)
    # Melakukan Tokenisasi dan Padding pada data teks
    prediction_input = token.texts_to_sequences(texts_p)
    # print(prediction_input)
    prediction_input = np.array(prediction_input).reshape(-1)
    # print(prediction_input)
    prediction_input = pad_sequences([prediction_input], maxlen=input_shape)
    # print(prediction_input)
    # Mendapatkan hasil prediksi keluaran pada model
    output = model.predict(prediction_input)
    output = output.argmax()
    # Menemukan respon sesuai data tag dan memainkan suara bot
    response_tag = encoder.inverse_transform([output])[0]
    response_text = random.choice(responses[response_tag])
    return jsonify({
        'response': str(response_text)
    })


if __name__ == '__main__':
    app.run(debug=False)
