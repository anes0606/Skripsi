import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Setup NLTK
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing Tools
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))
negation_words = {'tidak', 'bukan', 'jangan', 'belum', 'kurang', 'ga', 'gak', 'tak', 'susah'}
custom_stopwords = stop_words - negation_words

def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

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
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = combine_negations(tokens)
    tokens = [t for t in tokens if t not in custom_stopwords]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

st.title("🔍 Sentiment Analysis with SVM")
st.markdown("Upload CSV dengan kolom `Ulasan` dan `Label`")

uploaded_file = st.file_uploader("Unggah Dataset", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'Ulasan' not in df.columns or 'Label' not in df.columns:
        st.error("File harus memiliki kolom 'Ulasan' dan 'Label'")
    else:
        df.dropna(subset=['Ulasan', 'Label'], inplace=True)
        df['preprocessed'] = df['Ulasan'].apply(preprocess_text)

        tfidf = TfidfVectorizer(token_pattern=r'\b[a-zA-Z]{2,}\b', ngram_range=(1,2))
        X = tfidf.fit_transform(df['preprocessed'])
        y = df['Label']

        smote = SMOTE(sampling_strategy='minority', random_state=42)
        X_bal, y_bal = smote.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

        param_grid = [
            {'kernel': ['linear'], 'C': [0.5, 1, 2]},
            {'kernel': ['rbf'], 'C': [0.5, 1, 2], 'gamma': [0.5, 1, 2]},
        ]

        st.info("Melatih model dengan GridSearchCV...")
        grid = GridSearchCV(SVC(probability=True), param_grid, cv=3, scoring='accuracy')
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        y_pred = model.predict(X_test)

        st.success("Model telah dilatih!")
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        st.subheader("Prediksi Manual")
        user_input = st.text_area("Masukkan ulasan untuk diprediksi")
        if st.button("Prediksi"):
            processed = preprocess_text(user_input)
            vector = tfidf.transform([processed])
            prediction = model.predict(vector)[0]
            proba = model.predict_proba(vector)[0]
            if prediction == 1:
                st.success(f"✅ Sentimen Positif ({proba[1]*100:.2f}%)")
            else:
                st.error(f"❌ Sentimen Negatif ({proba[0]*100:.2f}%)")
