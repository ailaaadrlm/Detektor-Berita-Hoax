import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Konfigurasi halaman ──────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Berita Hoaks",
    page_icon="🔍",
    layout="centered"
)

# ── Load model dari HuggingFace Hub ─────────────────────────────
MODEL_NAME = "ailaadrlm/detekto-berita-hoax-indobert" 

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

# ── Fungsi preprocessing ─────────────────────────────────────────
def preprocess_text(text, max_words=300):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words])
    return text

# ── Fungsi prediksi ──────────────────────────────────────────────
def predict(text, tokenizer, model):
    clean = preprocess_text(text)
    inputs = tokenizer(
        clean,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding="max_length"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).squeeze()
    prob_asli  = probs[0].item()
    prob_hoaks = probs[1].item()
    label = "HOAKS" if prob_hoaks > prob_asli else "ASLI"
    return label, prob_asli, prob_hoaks

# ── Fungsi visualisasi probabilitas ─────────────────────────────
def plot_probabilitas(prob_asli, prob_hoaks):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    kategori = ['Berita Asli', 'Berita Hoaks']
    nilai    = [prob_asli * 100, prob_hoaks * 100]
    warna    = ['#2ecc71', '#e74c3c']
    bars = ax.barh(kategori, nilai, color=warna, height=0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probabilitas (%)')
    ax.set_title('Hasil Prediksi Model')
    for bar, val in zip(bars, nilai):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', fontweight='bold'
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

# ── Tampilan utama ───────────────────────────────────────────────
st.title("🔍 Deteksi Berita Hoaks")
st.markdown("Masukkan teks berita di bawah ini untuk mengecek apakah berita tersebut **asli** atau **hoaks**.")
st.divider()

# Load model
with st.spinner("Memuat model IndoBERT..."):
    tokenizer, model = load_model()

st.success("Model siap digunakan!")

# Input teks
teks_input = st.text_area(
    label="Teks Berita",
    placeholder="Tempel teks berita di sini...",
    height=200
)

# Tombol prediksi
if st.button("🔎 Cek Sekarang", type="primary", use_container_width=True):
    if not teks_input.strip():
        st.warning("Teks berita tidak boleh kosong.")
    elif len(teks_input.strip().split()) < 5:
        st.warning("Teks terlalu pendek. Masukkan minimal 5 kata.")
    else:
        with st.spinner("Menganalisis teks..."):
            label, prob_asli, prob_hoaks = predict(teks_input, tokenizer, model)

        st.divider()

        # Tampilkan hasil
        if label == "HOAKS":
            st.error(f"## ⚠️ {label}", icon="🚨")
            st.markdown("Model mendeteksi bahwa berita ini kemungkinan besar adalah **hoaks**.")
        else:
            st.success(f"## ✅ {label}", icon="✅")
            st.markdown("Model mendeteksi bahwa berita ini kemungkinan besar adalah **berita asli**.")

        # Visualisasi probabilitas
        st.markdown("### Probabilitas Prediksi")
        fig = plot_probabilitas(prob_asli, prob_hoaks)
        st.pyplot(fig)

        # Metrik detail
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Probabilitas Asli",
                value=f"{prob_asli*100:.1f}%"
            )
        with col2:
            st.metric(
                label="Probabilitas Hoaks",
                value=f"{prob_hoaks*100:.1f}%"
            )

        st.divider()
        st.caption("*Hasil prediksi berdasarkan model IndoBERT yang dilatih pada dataset berita Indonesia.*")
