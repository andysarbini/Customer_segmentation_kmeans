import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.ticker as ticker

# Judul aplikasi
st.title("Dashboard Segmentasi Customer Perusahaan Pembiayaan Kendaraan")

# Load data
@st.cache_data
def load_data():
  return pd.read_csv("data_customer.csv")

df = load_data()
st.subheader("Data Customer (Sample)")
st.dataframe(df.head())

# Visualisasi distribusi usia
st.subheader("Distribusi Usia Customer")
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df['usia'], bins=10, kde=True, ax=ax)
plt.xlabel("Usia")
st.pyplot(fig)

# Standarisasi fitur numerik
features = ['usia', 'penghasilan_bulanan', 'skor_kredit', 'lama_menjadi_nasabah']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_data)

# Visualisasi hasil cluster (Usia vs Penghasilan)
st.subheader("Visualisasi Cluster (Usia vs Penghasilan)")
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=df, x='usia', y='penghasilan_bulanan', hue='cluster', palette='Set2', ax=ax2)
plt.xlabel("Usia")
plt.ylabel("Penghasilan")
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
st.pyplot(fig2)

# Visualisasi hasil cluster (PCA 2D)
st.subheader("Visualisasi Cluster dengan PCA")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# Hitung rata-rata profil per cluster
cluster_means = df.groupby('cluster')[features].mean().round(2).reset_index()

fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='cluster', palette='Set2', ax=ax3)

# Tambahkan label rata-rata ke plot
for _, row in cluster_means.iterrows():
    cluster_center = df[df['cluster'] == row['cluster']][['PCA1', 'PCA2']].mean()
    label = f"Cluster {int(row['cluster'])}\nUsia: {row['usia']}\nGaji: {row['penghasilan_bulanan']/1e6:.1f} jt\nSkor: {row['skor_kredit']}\nLama: {row['lama_menjadi_nasabah']}"
    ax3.text(cluster_center['PCA1'], cluster_center['PCA2'], label, fontsize=8, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))

plt.title("Visualisasi Klaster dengan PCA")
st.pyplot(fig3)

# Rangkuman cluster
st.subheader("Rangkuman Segmentasi Customer")
st.write(df.groupby('cluster')[features].mean())

# Interpretasi PCA dan pengaruh fitur
st.subheader("Interpretasi PCA")
pca_components = pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2'])
st.write("PCA mengubah fitur asli menjadi dua komponen utama (PC1 dan PC2) yang merupakan kombinasi linear dari fitur awal. Komponen ini menyimpan informasi variasi terbesar dalam data.")
st.write("Berikut adalah kontribusi masing-masing fitur terhadap komponen utama:")
st.dataframe(pca_components.T.style.format("{:.2f}"))

# Rangkuman cluster
st.subheader("Rangkuman Segmentasi Customer")
st.write(df.groupby('cluster')[features].mean())

# Rekomendasi dan Ringkasan
st.subheader("📌 Rekomendasi Strategi per Cluster")
st.write("Berikut adalah ringkasan rekomendasi strategi pemasaran berdasarkan segmentasi customer:")

recommendation_data = {
    'Cluster': [0, 1, 2],
    'Karakteristik Utama': [
        "Usia menengah, penghasilan rendah, loyalitas tinggi",
        "Usia lebih tua, penghasilan menengah, loyalitas rendah",
        "Usia muda, penghasilan tinggi, loyalitas sedang"
    ],
    'Strategi Rekomendasi': [
        "Berikan apresiasi loyalitas berupa program referral atau diskon khusus",
        "Dorong loyalitas dengan promo pendaftaran dan layanan yang mudah",
        "Tawarkan program eksklusif berbasis benefit (mis. reward points, cashback premium)"
    ]
}

# Saran Segmentasi
st.subheader("Saran Segmentasi untuk Promo")
st.markdown("""
- **Cluster 0**: Usia menengah, penghasilan rendah, loyalitas tinggi - cocok untuk **Berikan apresiasi loyalitas berupa program referral atau diskon khusus**.
- **Cluster 1**: Usia lebih tua, penghasilan menengah, loyalitas rendah **Dorong loyalitas dengan promo pendaftaran dan layanan yang mudah**.
- **Cluster 2**: Usia muda, penghasilan tinggi, loyalitas sedang **Tawarkan program eksklusif berbasis benefit (mis. reward points, cashback premium)**.
""")

recommendation_df = pd.DataFrame(recommendation_data)
st.dataframe(recommendation_df)

