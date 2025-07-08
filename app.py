import streamlit as st
import numpy as np
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import streamlit.components.v1 as components

# Preprocessing functions
def case_folding(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub(r"[^\w\s]", "", text)

factory = StemmerFactory()
stemmer_engine = factory.create_stemmer()

def stemmer(text):
    return stemmer_engine.stem(text)

# Load model
def load_model():
    with open('model_svm2.pkl', 'rb') as f:
        return pickle.load(f)

model_package = load_model()
model = model_package['model']
tfidf = model_package['tfidf_vectorizer']

# App layout
st.title('🔍 Analisis Sentimen Ulasan Produk')
st.sidebar.success('🚀 Selamat datang di aplikasi analisis sentimen!')
st.sidebar.info('''
Aplikasi ini menggunakan model Support Vector Machine (SVM) untuk mengklasifikasikan
sentimen dari ulasan produk secara otomatis.
''')
st.sidebar.header('Informasi Model Terbaik')
st.sidebar.write('*Kernel:* Sigmoid')
st.sidebar.write('*Accuracy:* 89%')

# Tabs
tab1, tab2 = st.tabs(["Prediksi", "Hasil Evaluasi Kernel"])

with tab1:
    user_input = st.text_area("Masukkan ulasan produk:", "Produk ini sangat bagus dan berkualitas...")
    if st.button('Prediksi Sentimen'):
        raw = user_input.strip().lower()
        # Aturan double negasi dan deteksi kata positif/negatif
        if 'tidak tidak' in raw:
            pred = 1
            note = ' (double negasi terdeteksi)'
        elif 'positif' in raw:
            pred = 1
            note = ' (kata "positif" terdeteksi)'
        elif 'negatif' in raw:
            pred = 0
            note = ' (kata "negatif" terdeteksi)'
        else:
            # Proses normal dengan model
            processed = stemmer(remove_punctuation(case_folding(user_input)))
            vec = tfidf.transform([processed]).toarray()
            pred = model.predict(vec)[0]
            note = ''

        st.subheader('Hasil Prediksi:')
        if pred == 1:
            st.success(f'✅ Sentimen Positif{note}')
        else:
            st.error(f'❌ Sentimen Negatif{note}')

with tab2:
    st.subheader('Perbandingan Data 90% : 10%')
    html = '''
    <style>
      table {border-collapse: collapse; width: 100%;}
      th, td {border: 1px solid #000; padding: 6px; text-align: center;}
      th {background-color: #dae8fc;}
      .group {font-weight: bold; text-align: left; background-color: #f2f2f2;}
      .param {text-align: left;}
    </style>
    <table>
      <tr>
        <th>Kernel</th>
        <th>Parameter</th>
        <th colspan="4">Nilai</th>
      </tr>
      <tr><td class="group" colspan="6">Kombinasi Parameter Pertama</td></tr>
      <tr><td>Linear</td><td class="param">C = 2</td><td></td><td></td><td></td><td></td></tr>
      <tr><td>RBF</td><td class="param">C = 2<br>Gamma = 0.5</td><td></td><td></td><td></td><td></td></tr>
      <tr><td>Polynomial</td><td class="param">C = 2<br>Degree = 2<br>Gamma = 1</td><td></td><td></td><td></td><td></td></tr>
      <tr><td>Sigmoid</td><td class="param">C = 2<br>Gamma = 0.5</td><td></td><td></td><td></td><td></td></tr>
      <tr><td class="group" colspan="6">Kombinasi Parameter Kedua</td></tr>
      <tr><td>Linear</td><td class="param">C = 1.5</td><td></td><td></td><td></td><td></td></tr>
      <tr><td>RBF</td<<td class="param">C = 1.5<br>Gamma = 0.5</td><td></td><td></td><td></td><td></td></tr>
      <tr><td>Polynomial</td><td class="param">C = 0.5<br>Degree = 2<br>Gamma = 1</td><td></td><td></td><td></td><td></td></tr>
      <tr><td>Sigmoid</td><td class="param">C = 1.5<br>Gamma = 1</td><td></td><td></td><td></td><td></td></tr>
      <tr><td class="group" colspan="6">Kombinasi Parameter Ketiga</td></tr>
      <tr><td>Linear</td><td class="param">C = 1</td><td></td><td></td><td></td><td></td></tr>
      <tr><td>RBF</td><td class="param">C = 2<br>Gamma = 1</td><td></td><td></td><td></td><td></td></tr>
      <tr><td>Polynomial</td><td class="param">C = 1.5<br>Degree = 2<br>Gamma = 1</td><td></td><td></td><td></td><td></td></tr>
      <tr><td>Sigmoid</td><td class="param">C = 2<br>Gamma = 1</td><td></td><td></td><td></td><td></td></tr>
      <tr><td class="group" colspan="6">Kombinasi Parameter Keempat</td></tr>
      <tr><td>Linear</td><td class="param">C = 0.5</td><td></td><td></td><td></td><td></td></tr>
      <tr><td>RBF</td><td class="param">C = 1<br>Gamma = 0.5</td><td></td><td></td><td></td><td></td></tr>
      <tr><td>Polynomial</td><td class="param">C = 1<br>Degree = 2<br>Gamma = 2</td><td></td><td></td><td></td><td></td></tr>
      <tr><td>Sigmoid</td><td class="param">C = 1<br>Gamma = 1</td><td></td><td></td><td></td><td></td></tr>
    </table>
    '''
    components.html(html, height=600)

# Footer
st.markdown("""
---
Aplikasi Analisis Sentimen - Dibangun dengan Streamlit & Scikit-learn
""")
