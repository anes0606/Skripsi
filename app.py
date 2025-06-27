import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==============================
# Fungsi Preprocessing
# ==============================

def case_folding(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

factory = StemmerFactory()
stemmer_engine = factory.create_stemmer()

def stemmer(text):
    return stemmer_engine.stem(text)

# ==============================
# Judul Aplikasi
# ==============================
st.title('🔍 Analisis Sentimen Ulasan Produk')
st.write('Aplikasi ini memprediksi sentimen ulasan (positif/negatif) menggunakan model SVM')

# Sidebar untuk informasi
st.sidebar.header('Tentang Aplikasi')
st.sidebar.info('''
Aplikasi ini menggunakan model Machine Learning (SVM) 
yang telah dilatih untuk mengklasifikasikan sentimen ulasan produk.
''')

# Fungsi untuk memuat model
@st.cache_resource
import gdown

url = "https://drive.google.com/uc?export=download&id=1tuqMY82MmriSSjYheUD64AZINr5PXlle"
output = "model_svm.pkl"
gdown.download(url, output, quiet=False)

# Memuat model
try:
    model_package = load_model()
    model = model_package.get('model', None)
    tfidf = model_package.get('tfidf_vectorizer', None)
    
    # Mengambil data tambahan dengan default value jika tidak tersedia
    best_model_metrics = model_package.get('best_model_metrics', {})
    dataset_info = model_package.get('dataset_info', {})
    kernel_comparison = model_package.get('kernel_comparison', {})
    classification_report = model_package.get('classification_report', {})

    # Menampilkan info model di sidebar jika tersedia
    st.sidebar.subheader('Informasi Model')
    if best_model_metrics:
        st.sidebar.write(f"Kernel: {best_model_metrics.get('kernel', 'N/A')}")
        st.sidebar.write(f"Akurasi: {best_model_metrics.get('accuracy', 0):.2f}")
    else:
        st.sidebar.warning("Informasi metrik model tidak tersedia.")

except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan 'model_svm.pkl' ada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {str(e)}")
    st.stop()

# Tab untuk navigasi
tab1, tab2, tab3 = st.tabs(["Prediksi", "Parameter Model", "Info Dataset"])

with tab1:
    user_input = st.text_area("Masukkan ulasan :", "Produk ini sangat bagus dan berkualitas...")

    if st.button('Prediksi Sentimen'):
        if model and tfidf:
            # Preprocessing input
            processed_input = case_folding(user_input)
            processed_input = remove_punctuation(processed_input)
            processed_input = stemmer(processed_input)

            # Transformasi TF-IDF dan konversi ke array dense
            input_vector = tfidf.transform([processed_input]).toarray()

            # Prediksi
            prediction = model.predict(input_vector)[0]

            # Tampilkan hasil
            st.subheader('Hasil Prediksi:')
            if prediction == 1:
                st.success('✅ Sentimen Positif')
                st.balloons()
            else:
                st.error('❌ Sentimen Negatif')

with tab2:
    st.subheader('Parameter Model Terbaik')
    
    # Jika parameter tidak tersedia, tampilkan contoh
    if not best_model_metrics or 'best_params' not in best_model_metrics:
        st.warning("Parameter terbaik tidak tersedia dalam model.")
        
        # Contoh parameter default untuk demonstrasi
        example_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'degree': 3,  # untuk kernel poly
            'coef0': 0.0  # untuk kernel poly/sigmoid
        }
        
        st.info("""
        **Contoh Parameter SVM yang Umum Digunakan:**
        (Ini hanya contoh untuk demonstrasi)
        """)
        st.json(example_params)
        
        # Visualisasi pengaruh parameter
        st.markdown("**Pengaruh Parameter C dan Gamma**")
        st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_rbf_parameters_001.png", 
                caption="Pengaruh parameter C dan gamma pada decision boundary (Sumber: scikit-learn)")
        
        st.markdown("""
        **Penjelasan Parameter:**
        - `kernel`: Jenis kernel (linear, rbf, poly, sigmoid)
        - `C`: Parameter regularisasi (nilai besar = kurang regularisasi)
        - `gamma`: Pengaruh satu contoh training (hanya untuk rbf, poly, sigmoid)
        - `degree`: Derajat untuk kernel polynomial
        """)
    else:
        st.json(best_model_metrics['best_params'])
        
        # Visualisasi parameter jika tersedia
        if 'C' in best_model_metrics['best_params'] and 'gamma' in best_model_metrics['best_params']:
            st.markdown("**Visualisasi Parameter**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Parameter C", best_model_metrics['best_params']['C'])
            with col2:
                st.metric("Parameter Gamma", str(best_model_metrics['best_params']['gamma']))

    st.subheader('Perbandingan Kernel')
    
    if not kernel_comparison:
        st.warning("Data perbandingan kernel tidak tersedia dalam model.")
        
        # Contoh data dummy untuk demonstrasi
        dummy_kernel_data = {
            'linear': {'accuracy': 0.82, 'precision': 0.83, 'recall': 0.81, 'f1': 0.82, 'fit_time': 1.5},
            'rbf': {'accuracy': 0.85, 'precision': 0.86, 'recall': 0.84, 'f1': 0.85, 'fit_time': 2.8},
            'poly': {'accuracy': 0.83, 'precision': 0.84, 'recall': 0.82, 'f1': 0.83, 'fit_time': 3.5}
        }
        
        st.info("""
        **Contoh Perbandingan Kernel:**
        (Data dummy untuk demonstrasi)
        """)
        
        kernel_df = pd.DataFrame.from_dict(dummy_kernel_data, orient='index')
        st.dataframe(kernel_df.style.format({
            'accuracy': '{:.2%}',
            'precision': '{:.2%}',
            'recall': '{:.2%}',
            'f1': '{:.2%}',
            'fit_time': '{:.2f} detik'
        }))
        
        # Visualisasi interaktif perbandingan
        st.markdown("**Visualisasi Interaktif Perbandingan Kernel**")
        metric_options = ['accuracy', 'precision', 'recall', 'f1', 'fit_time']
        selected_metric = st.selectbox("Pilih metrik untuk visualisasi:", metric_options)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        kernel_df[selected_metric].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_title(f'Perbandingan {selected_metric.capitalize()} antar Kernel')
        ax.set_ylabel(selected_metric)
        
        if selected_metric != 'fit_time':
            ax.set_ylim(0.7, 0.9)  # Untuk metrik akurasi/precision/recall/f1
            ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.3)
        else:
            ax.set_ylabel('Waktu (detik)')
        
        st.pyplot(fig)
        
        st.markdown("""
        **Penjelasan:**
        - **Linear Kernel**: Cepat dan baik untuk data yang dapat dipisahkan secara linear
        - **RBF Kernel**: Fleksibel untuk data kompleks, tetapi lebih lambat
        - **Polynomial Kernel**: Cocok untuk hubungan non-linear yang teratur
        """)
    else:
        kernel_df = pd.DataFrame.from_dict(kernel_comparison, orient='index')
        st.dataframe(kernel_df.style.format({
            'accuracy': '{:.2%}',
            'precision': '{:.2%}',
            'recall': '{:.2%}',
            'f1': '{:.2%}',
            'fit_time': '{:.2f} detik'
        }))
        
        # Visualisasi jika data tersedia
        st.markdown("**Visualisasi Performa Kernel**")
        tab1, tab2, tab3 = st.tabs(["Akurasi", "Waktu Training", "F1-Score"])
        
        with tab1:
            fig1, ax1 = plt.subplots()
            kernel_df['accuracy'].plot(kind='bar', ax=ax1, color='green', alpha=0.6)
            ax1.set_title('Akurasi Berbagai Kernel')
            ax1.set_ylabel('Akurasi')
            st.pyplot(fig1)
        
        with tab2:
            fig2, ax2 = plt.subplots()
            kernel_df['fit_time'].plot(kind='bar', ax=ax2, color='blue', alpha=0.6)
            ax2.set_title('Waktu Training Berbagai Kernel')
            ax2.set_ylabel('Detik')
            st.pyplot(fig2)
        
        with tab3:
            fig3, ax3 = plt.subplots()
            kernel_df['f1'].plot(kind='bar', ax=ax3, color='purple', alpha=0.6)
            ax3.set_title('F1-Score Berbagai Kernel')
            ax3.set_ylabel('F1-Score')
            st.pyplot(fig3)

    st.subheader('Laporan Klasifikasi')
    if classification_report:
        # Jika classification_report adalah string
        if isinstance(classification_report, str):
            st.text(classification_report)
        # Jika classification_report adalah dictionary (output dari classification_report dengan output_dict=True)
        elif isinstance(classification_report, dict):
            report_df = pd.DataFrame(classification_report).transpose()
            
            # Hapus kolom support jika tidak diperlukan
            if 'support' in report_df.columns:
                report_df = report_df.drop(columns=['support'])
            
            st.dataframe(report_df.style.format("{:.2f}").background_gradient(cmap='Blues'))
            
            # Visualisasi metrik
            st.markdown("**Visualisasi Metrik Klasifikasi**")
            metrics_to_plot = report_df.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
            
            fig, ax = plt.subplots(figsize=(8, 4))
            metrics_to_plot[['precision', 'recall', 'f1-score']].plot(
                kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
            ax.set_title('Precision, Recall, dan F1-Score per Kelas')
            ax.set_ylabel('Nilai')
            ax.set_ylim(0, 1.1)
            st.pyplot(fig)
    else:
        st.warning("Laporan klasifikasi tidak tersedia.")
        
        # Contoh laporan klasifikasi dummy
        dummy_report = {
            'negatif': {'precision': 0.85, 'recall': 0.80, 'f1-score': 0.82, 'support': 500},
            'positif': {'precision': 0.83, 'recall': 0.87, 'f1-score': 0.85, 'support': 500},
            'accuracy': 0.84,
            'macro avg': {'precision': 0.84, 'recall': 0.835, 'f1-score': 0.835, 'support': 1000},
            'weighted avg': {'precision': 0.84, 'recall': 0.84, 'f1-score': 0.84, 'support': 1000}
        }
        
        st.info("""
        **Contoh Laporan Klasifikasi:**
        (Data dummy untuk demonstrasi)
        """)
        
        report_df = pd.DataFrame(dummy_report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))

with tab3:
    st.subheader('Informasi Dataset')
    
    if not dataset_info:
        st.warning("Informasi dataset tidak tersedia dalam model.")
        
        # Contoh data dummy
        dummy_dataset_info = {
            'num_samples': 2000,
            'class_distribution': {'positif': 1200, 'negatif': 800},
            'text_examples': {
                'positif': [
                    "Produk ini sangat bagus kualitasnya",
                    "Sangat puas dengan pembelian ini",
                    "Pengiriman cepat dan produk sesuai gambar"
                ],
                'negatif': [
                    "Kualitas produk tidak sesuai harapan",
                    "Pengiriman sangat lambat",
                    "Barang rusak saat diterima"
                ]
            },
            'text_length_stats': {
                'avg_length': 45,
                'min_length': 5,
                'max_length': 120
            }
        }
        
        st.info("""
        **Contoh Informasi Dataset:**
        (Data dummy untuk demonstrasi)
        """)
        
        dataset_info = dummy_dataset_info
    
    # Tampilkan informasi dataset
    st.write(f"**Jumlah total sampel:** {dataset_info.get('num_samples', 'N/A')}")
    
    # Visualisasi distribusi kelas
    class_dist = dataset_info.get('class_distribution', {})
    if class_dist:
        st.markdown("### Distribusi Kelas")
        col1, col2 = st.columns(2)
        
        with col1:
            dist_df = pd.DataFrame.from_dict(class_dist, orient='index', columns=['Jumlah'])
            dist_df['Persentase'] = (dist_df['Jumlah'] / dist_df['Jumlah'].sum()) * 100
            st.dataframe(dist_df.style.format({'Persentase': '{:.1f}%'}))
        
        with col2:
            fig, ax = plt.subplots()
            dist_df['Jumlah'].plot.pie(autopct='%1.1f%%', ax=ax, 
                                     colors=['#ff9999','#66b3ff'], 
                                     labels=dist_df.index)
            ax.set_ylabel('')
            st.pyplot(fig)
    
    # Statistik panjang teks jika tersedia
    if 'text_length_stats' in dataset_info:
        st.markdown("### Statistik Panjang Teks")
        stats = dataset_info['text_length_stats']
        col1, col2, col3 = st.columns(3)
        col1.metric("Rata-rata Panjang", f"{stats['avg_length']} karakter")
        col2.metric("Terkecil", f"{stats['min_length']} karakter")
        col3.metric("Terpanjang", f"{stats['max_length']} karakter")
        
        # Generate dummy length distribution
        st.markdown("**Distribusi Panjang Teks**")
        np.random.seed(42)
        dummy_lengths = np.random.normal(stats['avg_length'], 15, 1000)
        dummy_lengths = np.clip(dummy_lengths, stats['min_length'], stats['max_length'])
        
        fig, ax = plt.subplots()
        ax.hist(dummy_lengths, bins=20, color='skyblue', edgecolor='black')
        ax.set_xlabel('Panjang Teks (karakter)')
        ax.set_ylabel('Frekuensi')
        ax.set_title('Distribusi Panjang Ulasan')
        st.pyplot(fig)
    
    # Contoh teks
    st.markdown("### Contoh Ulasan")
    text_examples = dataset_info.get('text_examples', {})
    
    if text_examples:
        tab1, tab2 = st.tabs(["Ulasan Positif", "Ulasan Negatif"])
        
        with tab1:
            if 'positif' in text_examples and text_examples['positif']:
                for example in text_examples['positif'][:5]:
                    st.success(f"👍 {example}")
            else:
                st.warning("Contoh ulasan positif tidak tersedia.")
        
        with tab2:
            if 'negatif' in text_examples and text_examples['negatif']:
                for example in text_examples['negatif'][:5]:
                    st.error(f"👎 {example}")
            else:
                st.warning("Contoh ulasan negatif tidak tersedia.")
    else:
        st.warning("Contoh ulasan tidak tersedia.")

# Catatan footer
st.markdown("---")
st.caption("""
Aplikasi Analisis Sentimen - Dibangun dengan Streamlit dan Scikit-learn  
© 2023 - Dibangun untuk demonstrasi klasifikasi teks
""")

# Tambahkan beberapa styling
st.markdown("""
<style>
    .stMetric {
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 10px;
        background-color: #f6f8fa;
    }
    .stDataFrame {
        border-radius: 8px;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)
