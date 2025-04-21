# Customer_segmentation_kmeans

# 🚗 Customer Segmentation with KMeans, Studi kasus Perusahaan Pembiayaan Kendaraan
Dashboard interaktif berbasis Streamlit untuk menganalisis dan memvisualisasikan segmentasi customer pada perusahaan pembiayaan mobil. Tujuan dari project ini adalah membantu perusahaan dalam menentukan strategi pemasaran seperti promo kredit berdasarkan karakteristik pelanggan.

## 📊 Fitur Dashboard

- Visualisasi eksploratif customer berdasarkan usia, penghasilan, skor kredit, dan lama menjadi nasabah.
- Segmentasi customer menggunakan K-Means Clustering.
- Tampilan hasil cluster dengan ringkasan karakteristik masing-masing segmen.
- Insight berbasis data untuk strategi penawaran kredit kendaraan.

## 🗂 Struktur Folder

```
dashboard-penjualan/
├── app.py                  # Aplikasi utama Streamlit
├── data_customer.csv           # Data penjualan contoh (opsional)
├── requirements.txt        # Dependensi untuk deployment
└── README.md               # Dokumen ini
```


## 🧪 Teknologi yang Digunakan

- Python
- Pandas
- Scikit-learn
- Matplotlib & Seaborn
- Streamlit

## 🚀 Cara Menjalankan Lokal

1. Clone repo ini:
   ```bash
   git clone https://github.com/username/nama-repo.git
   cd nama-repo

## 💻 Cara Menjalankan Lokal

1. Clone repo ini:
    ```bash
    git clone https://github.com/username/nama-repo.git
    cd dashboard-penjualan
    ```

2. Install dependensi:
    ```bash
    pip install -r requirements.txt
    ```

3. Jalankan Streamlit:
    ```bash
    streamlit run app.py
    ```

## ☁️ Deploy ke Streamlit Cloud

1. Upload semua file ke GitHub
2. Buka [streamlit.io/cloud](https://streamlit.io/cloud)
3. Klik **“New App”**, pilih repo ini
4. Pilih file `app.py` lalu klik **Deploy**

---

🧑‍💻 Dibuat oleh: Andy Sarbini  
📬 Kontak: kangandy09@gmail.com
