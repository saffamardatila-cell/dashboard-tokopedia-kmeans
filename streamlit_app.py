import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard Keluhan Tokopedia - K-Means", layout="wide")
st.title("Analisis Pengelompokan Keluhan Pengguna Aplikasi Tokopedia")
st.caption("Metode: TF-IDF + K-Means Clustering (K=3) | Daffa Putra Hartanto - 2270065160054 | Universitas Nasional")

clusters = {
    "Cluster 0 - Promo, Voucher & Diskon": {
        "jumlah": 1000, "persen": 16.76,
        "kata": ["promo", "voucher", "diskon", "gagal", "klaim", "flash sale"],
        "desc": "Keluhan promo, voucher & diskon tidak dapat digunakan.",
        "contoh": ["Voucher promo gagal dipakai padahal masih berlaku", "Diskon flash sale tidak bisa diklaim", "Kode promo selalu gagal saat checkout"]
    },
    "Cluster 1 - Barang / Belanja": {
        "jumlah": 4077, "persen": 68.35,
       "kata": ["barang","belanja","harga","toko","produk","kualitas","bagus","sesuai","pesan","pengiriman"],
        "desc": "Keluhan seputar barang, harga, kualitas produk, dan pengalaman belanja.",
        "contoh": "Barang tidak sesuai deskripsi, harga mahal, kualitas jelek.",
    },
    "Cluster 2 - Aplikasi": {
        "jumlah": 888, "persen": 14.89,
       "kata": ["aplikasi","error","login","update","bug","lemot","akun","keluar","buka","gagal"],
       "desc": "Keluhan seputar error, login, dan kendala teknis aplikasi.",
        "contoh": "Aplikasi error, tidak bisa login, lemot setelah update.",
    },
}

def prediksi_cluster(teks):
    t = teks.lower()
    skor = {}
    for nama, info in clusters.items():
        skor[nama] = sum(1 for k in info["kata"] if k.lower() in t)
    best = max(skor, key=skor.get)
    if skor[best] == 0:
     return "Cluster 2 - Aplikasi", 0
    return best, skor[best]

st.sidebar.header("Filter")
pilihan = st.sidebar.multiselect("Pilih Klaster", list(clusters.keys()), default=list(clusters.keys()))
keyword = st.sidebar.text_input("Cari kata kunci / ulasan", "").lower()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ulasan Mentah", "6.316")
c2.metric("Ulasan Bersih", "5.770")
c3.metric("Ulasan Terbuang", "546")
c4.metric("Jumlah Klaster (K)", "3")
st.divider()

st.subheader("Proporsi & Sebaran Klaster")
df = pd.DataFrame([{"Klaster": k, "Jumlah": v["jumlah"], "Persentase": v["persen"]} for k, v in clusters.items()])
df = df[df["Klaster"].isin(pilihan)]
st.bar_chart(df.set_index("Klaster")["Jumlah"])
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Karakteristik Klaster & Kata Kunci Dominan")
for nama in pilihan:
    info = clusters[nama]
    with st.expander(f"{nama} - {info['jumlah']} ulasan ({info['persen']}%)"):
        st.write(info["desc"])
        st.write("**Kata kunci:** " + ", ".join(info["kata"]))
        st.write("**Contoh ulasan:**")
        for c in info["contoh"]:
            if not keyword or keyword in c.lower():
                st.write(f"- {c}")

st.subheader("Pencarian Ulasan (Contoh)")
if keyword:
    found = []
    for nama, info in clusters.items():
        for c in info["contoh"]:
            if keyword in c.lower() or keyword in " ".join(info["kata"]).lower():
                found.append({"Klaster": nama, "Ulasan": c})
    if found:
        st.dataframe(pd.DataFrame(found), use_container_width=True, hide_index=True)
    else:
        st.info("Tidak ditemukan di contoh. Coba kata: promo, ongkir, saldo, refund, voucher")
else:
    st.write("Ketik di sidebar, misal: promo / ongkir / saldo")

st.divider()
st.subheader("Coba Klasifikasi Ulasan Baru")
teks_baru = st.text_area("Tulis keluhan, contoh: voucher saya gagal dipakai", "")
if st.button("Prediksi Klaster"):
    if teks_baru.strip() == "":
        st.warning("Isi dulu ulasannya.")
    else:
        hasil, skor = prediksi_cluster(teks_baru)
        st.success(f"Hasil prediksi: **{hasil}**")
        st.write(clusters[hasil]["desc"])
        st.write("Kata kunci yang cocok: " + ", ".join([k for k in clusters[hasil]["kata"] if k.lower() in teks_baru.lower()]) or "-")

st.divider()
st.caption("Sumber data: Ulasan Google Play Store Tokopedia, 20 Juli - 5 November 2026. Preprocessing: case folding, cleaning, tokenizing, normalisasi, stopword removal, stemming Sastrawi. TF-IDF max_features=1000, K-Means n_clusters=3, random_state=42, n_init=10.")  
