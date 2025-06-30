import streamlit as st
import numpy as np
import pickle
import pandas as pd
import re
import gdown
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

# Preprocessing functions
def case_folding(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

factory = StemmerFactory()
stemmer_engine = factory.create_stemmer()

def stemmer(text):
    return stemmer_engine.stem(text)

# App title
st.title('🔍 Analisis Sentimen Ulasan')
st.write('Aplikasi ini memprediksi sentimen ulasan (positif/negatif) menggunakan model SVM')

# Sidebar Animation and Information
st.sidebar.success('🚀 Selamat datang di aplikasi analisis sentimen!')
st.sidebar.info('''
Aplikasi ini menggunakan model Support Vector Machine (SVM) untuk mengklasifikasikan
sentimen dari ulasan produk secara otomatis.
''')
st.sidebar.header('Informasi Model Terbaik')
st.sidebar.write('*Kernel:* Sigmoid')
st.sidebar.write('*Accuracy:* 89%')

# Load model
@st.cache_resource
def load_model():
    file_id = "1wS9BPwcr2ol5dYJ4k1msDMTKnG6FXxL6"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output_path = "model.pkl"

    try:
        response = requests.get(url)
        response.raise_for_status()  # cek kalau gagal ambil file

        # Simpan file ke lokal
        with open(output_path, "wb") as f:
            f.write(response.content)

        # Load model
        with open(output_path, "rb") as f:
            model = pickle.load(f)

        return model

    except Exception as e:
        raise RuntimeError(f"Gagal memuat model dari Google Drive: {e}")

model_package = load_model()
model = model_package.get('model')
tfidf = model_package.get('tfidf_vectorizer')

# Tabs
tab1, tab2 = st.tabs(["Prediksi", "Hasil Evaluasi Kernel"])

with tab1:
    user_input = st.text_area("Masukkan ulasan produk:", "Produk ini sangat bagus dan berkualitas...")

    if st.button('Prediksi Sentimen'):
        processed_input = stemmer(remove_punctuation(case_folding(user_input)))
        input_vector = tfidf.transform([processed_input]).toarray()
        prediction = model.predict(input_vector)[0]

        st.subheader('Hasil Prediksi:')
        if prediction == 1:
            st.success('✅ Sentimen Positif')
        else:
            st.error('❌ Sentimen Negatif')

with tab2:
    st.subheader('Perbandingan Kernel (Data 80% : 20%)')

    kernel_data = {
        'Kernel': ['Linear', 'RBF', 'Polynomial', 'Sigmoid'],
        'Parameter': ['C=2', 'C=2, gamma=0.5', 'C=2, degree=2, gamma=1', 'C=2, gamma=1'],
        'Accuracy': [0.87, 0.88, 0.83, 0.89],
        'Precision': [0.87, 0.90, 0.88, 0.89],
        'Recall': [0.81, 0.82, 0.72, 0.84],
        'F1-Score': [0.84, 0.84, 0.75, 0.86]
    }

    kernel_df = pd.DataFrame(kernel_data)
    kernel_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']] = kernel_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].applymap(lambda x: f"{x:.0%}")

    st.table(kernel_df)

# Footer
st.markdown("""
---
Aplikasi Analisis Sentimen - Dibangun dengan Streamlit & Scikit-learn
""")
