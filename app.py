import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Judul aplikasi
st.title("Dashboard Segmentasi Customer Perusahaan Pembiayaan Kendaraan")

# Load data
@st.cache_data
def load_data():
  return pd.read_csv("data_customer.csv")

df = load_data()
st.subheader("Data Customer (Sample)")
st.dataframe(df.head())

# Standarisasi fitur numerik
fitur_numerik = ['usia', 'penghasilan_bulanan', 'skor_kredit', 'lama_menjadi_nasabah']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[fitur_numerik])

# K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled_features)

# Rata-rata tiap cluster
st.subheader("Rata-rata Tiap Cluster")
st.dataframe(df.groupby('cluster')[fitur_numerik].mean())

# Visualisasi PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
pca_df['cluster'] = df['cluster']

st.subheader("Visualisasi Cluster Customer (PCA)")
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='Set2', s=80, ax=ax)
ax.set_title("Visualisasi Cluster Customer dengan PCA")
st.pyplot(fig)

# Saran Segmentasi
st.subheader("Saran Segmentasi untuk Promo")
st.markdown("""
- **Cluster 0**: Customer loyal dengan masa menjadi nasabah panjang dan penghasilan sedang - cocok untuk **promo eksklusif loyalti**.
- **Cluster 1**: Penghasilan tinggi tapi masa nasabah masih pendek = bisa diberi **penawaran menarik untuk retensi**.
--**Cluster 2**: Skor kredit tinggi dan penghasilan menengah - cocok untuk **penawaran kredit cepat**.
--**Cluster 3**: Usia lebih tua, penghasilan rendah, belum lama menjadi nasabah - bisa diberi **edukasi atau penawaran ringan**.
""")
