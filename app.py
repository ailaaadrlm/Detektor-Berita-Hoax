import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import re
from transformers import AutoTokenizer, pipeline
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
MODEL_NAME = "ailaadrlm/detektor-berita-hoax-indobert"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    clf = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True,
        max_length=256,
    )
    return clf

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
def predict(text, clf):
    clean = preprocess_text(text)
    results = clf(clean)[0]
    scores = {r["label"]: r["score"] for r in results}

    prob_asli = scores.get(
        "LABEL_0", scores.get("asli", scores.get("Asli", scores.get("ASLI", 0.0)))
    )
    prob_hoaks = scores.get(
        "LABEL_1", scores.get("hoaks", scores.get("Hoaks", scores.get("HOAKS", 0.0)))
    )

    # Fallback jika nama label tidak dikenali
    if prob_asli == 0.0 and prob_hoaks == 0.0:
        all_scores = list(scores.values())
        prob_asli  = all_scores[0] if len(all_scores) > 0 else 0.5
        prob_hoaks = all_scores[1] if len(all_scores) > 1 else 0.5

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

with st.spinner("Memuat model IndoBERT..."):
    clf = load_model()

st.success("Model siap digunakan!")

teks_input = st.text_area(
    label="Teks Berita",
    placeholder="Tempel teks berita di sini...",
    height=200
)

if st.button("🔎 Cek Sekarang", type="primary", use_container_width=True):
    if not teks_input.strip():
        st.warning("Teks berita tidak boleh kosong.")
    elif len(teks_input.strip().split()) < 5:
        st.warning("Teks terlalu pendek. Masukkan minimal 5 kata.")
    else:
        with st.spinner("Menganalisis teks..."):
            label, prob_asli, prob_hoaks = predict(teks_input, clf)

        st.divider()

        if label == "HOAKS":
            st.error(f"## ⚠️ {label}", icon="🚨")
            st.markdown("Model mendeteksi bahwa berita ini kemungkinan besar adalah **hoaks**.")
        else:
            st.success(f"## ✅ {label}", icon="✅")
            st.markdown("Model mendeteksi bahwa berita ini kemungkinan besar adalah **berita asli**.")

        st.markdown("### Probabilitas Prediksi")
        fig = plot_probabilitas(prob_asli, prob_hoaks)
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Probabilitas Asli", value=f"{prob_asli*100:.1f}%")
        with col2:
            st.metric(label="Probabilitas Hoaks", value=f"{prob_hoaks*100:.1f}%")

        st.divider()
        st.caption("*Hasil prediksi berdasarkan model IndoBERT yang dilatih pada dataset berita Indonesia.*")
