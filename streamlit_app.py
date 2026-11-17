import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard Keluhan Tokopedia - K-Means", layout="wide")
st.title("Analisis Pengelompokan Keluhan Pengguna Aplikasi Tokopedia")
st.caption("Metode: TF-IDF + K-Means Clustering (K=3) | Daffa Putra Hartanto - 2270065160054 | Universitas Nasional")

clusters = {
    "Cluster 0 - Promo, Voucher & Diskon": {
        "jumlah": 937, "persen": 16.24,
        "kata": ["promo", "voucher", "diskon", "gagal", "klaim", "flash sale"],
        "desc": "Keluhan promo, voucher & diskon tidak dapat digunakan.",
        "contoh": ["Voucher promo gagal dipakai padahal masih berlaku", "Diskon flash sale tidak bisa diklaim", "Kode promo selalu gagal saat checkout"]
    },
    "Cluster 1 - Ongkir & Pengiriman": {
        "jumlah": 349, "persen": 6.05,
        "kata": ["ongkir", "biaya", "harga", "kirim", "kurir", "rusak", "terlambat"],
        "desc": "Keluhan biaya ongkir dan pengiriman.",
        "contoh": ["Ongkir mahal banget padahal dekat", "Kurir terlambat, paket rusak saat sampai", "Biaya kirim naik terus, harga jadi mahal"]
    },
    "Cluster 2 - Akun, Dana & Kendala Aplikasi": {
        "jumlah": 4484, "persen": 77.71,
        "kata": ["saldo", "top up", "verifikasi", "KTP", "dana", "refund"],
        "desc": "Keluhan akun, dana & kendala aplikasi. Klaster paling dominan.",
        "contoh": ["Saldo tidak masuk padahal sudah top up", "Verifikasi KTP gagal terus, dana tertahan", "Refund lama banget tidak kembali ke saldo"]
    },
}

def prediksi_cluster(teks):
    t = teks.lower()
    skor = {}
    for nama, info in clusters.items():
        skor[nama] = sum(1 for k in info["kata"] if k.lower() in t)
    best = max(skor, key=skor.get)
    if skor[best] == 0:
        return "Cluster 2 - Akun, Dana & Kendala Aplikasi", 0
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
