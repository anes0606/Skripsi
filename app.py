import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# === PREPARE NLP TOOLS ===
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))
negation_words = {'tidak', 'bukan', 'jangan', 'belum', 'kurang', 'ga', 'gak', 'tak', 'susah'}
custom_stopwords = stop_words - negation_words

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def case_folding(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def combine_negations(tokens):
    combined = []
    skip = False
    for i in range(len(tokens)):
        if skip:
            skip = False
            continue
        if tokens[i] in negation_words and i+1 < len(tokens):
            combined.append(f"{tokens[i]}_{tokens[i+1]}")
            skip = True
        else:
            combined.append(tokens[i])
    return combined

def preprocess_text(text):
    text = clean_text(text)
    text = case_folding(text)
    text = remove_punctuation(text)
    tokens = word_tokenize(text)
    tokens = combine_negations(tokens)
    tokens = [t for t in tokens if t not in custom_stopwords]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# === LOAD AND PREPARE DATA ===
# Untuk demo, data dummy
data = {
    'Ulasan': [
        'Produk ini sangat bagus dan berkualitas',
        'Saya tidak suka layanan yang diberikan',
        'Pengiriman cepat dan produk sesuai',
        'Barang jelek dan tidak sesuai deskripsi'
    ],
    'Label': [1, 0, 1, 0]
}
df = pd.DataFrame(data)

df['Cleaned'] = df['Ulasan'].apply(preprocess_text)

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['Cleaned'])
y = df['Label']

# Train model
model = SVC(kernel='linear', C=1)
model.fit(X, y)

# === STREAMLIT UI ===
st.set_page_config(page_title="Analisis Sentimen", layout="wide")

st.markdown("<h1 style='text-align: center;'>🔍 Analisis Sentimen Ulasan</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Aplikasi ini memprediksi sentimen ulasan (positif/negatif) menggunakan model SVM</p>", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Tentang Aplikasi")
    st.info("Aplikasi ini menggunakan model Machine Learning (SVM) yang telah dilatih untuk mengklasifikasikan sentimen ulasan produk.")

ulasan = st.text_area("Masukkan Ulasan")

if st.button("Prediksi"):
    if not ulasan.strip():
        st.warning("⚠️ Harap masukkan teks ulasan terlebih dahulu.")
    else:
        cleaned = preprocess_text(ulasan)
        vector = tfidf.transform([cleaned])
        hasil = model.predict(vector)[0]
        label = "Positif" if hasil == 1 else "Negatif"
        st.success(f"Hasil Prediksi Sentimen: {label}")
