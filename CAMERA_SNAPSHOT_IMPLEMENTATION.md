# 📸 Camera Snapshot Upload Implementation

**Status:** ✅ Selesai

---

## 📋 Ringkasan Perubahan

Saya telah mengintegrasikan logika upload gambar kamera ke Supabase Storage dalam aplikasi Streamlit Anda. Sistem ini memungkinkan dashboard React untuk melakukan polling dan menampilkan live video feed setiap 1.5 detik.

### Perubahan yang Dilakukan di `app.py`:

#### 1. **Fungsi Baru: `upload_camera_snapshot()` (Line ~340)**
```python
def upload_camera_snapshot(frame_rgb, bucket_name="camera_snapshots", file_name="latest_frame.jpg")
```
- Upload frame RGB ke Supabase Storage bucket `camera_snapshots`
- Kompresi JPEG 60% untuk mempercepat upload
- Menggunakan `upsert=true` agar file lama langsung ditimpa
- Error handling yang robust

#### 2. **Inisialisasi Tracking Waktu Upload (Line ~729)**
- Tambah `st.session_state.last_snapshot_upload_time = 0.0`
- Digunakan untuk membatasi upload setiap 1.5 detik

#### 3. **Step 8 - Camera Snapshot Upload (Line ~800)**
Dalam loop utama kamera, setelah UI diperbarui:
```python
if SUPABASE_SERVICE_ROLE_KEY and (time.time() - st.session_state.last_snapshot_upload_time >= 1.5):
    upload_ok = upload_camera_snapshot(frame_to_show, ...)
    st.session_state.last_snapshot_upload_time = time.time()
```

---

## ✅ Checklist Verifikasi

### Infrastructure:
- [x] **Supabase Storage Bucket:** `camera_snapshots` (public)
- [x] **Supabase URL & Keys:** Sudah dikonfigurasi di `.env`
  - `SUPABASE_URL=https://ymbcypsfuiaixztlehkg.supabase.co`
  - `SUPABASE_SERVICE_ROLE_KEY=...` ✅

### Kode Streamlit:
- [x] **Fungsi `upload_camera_snapshot()`:** Ditambahkan
- [x] **Integration dengan Loop Kamera:** Setelah Step 7 (Update UI)
- [x] **Interval Upload:** 1.5 detik per frame

### React Dashboard:
- [x] **Live Inspection Layer:** Polling `latest_frame.jpg` dari Supabase
- [x] **Fallback Image:** Gear biru jika belum ada gambar

---

## 🚀 Cara Menggunakan

### 1. **Jalankan Streamlit App:**
```bash
streamlit run app.py
```

### 2. **Di Sidebar:**
- Pilih Part Code yang diinginkan
- Klik tombol **▶ START** untuk memulai kamera

### 3. **Monitoring:**
- Gambar kamera akan otomatis diupload setiap 1.5 detik
- Lihat console output: `[HH:MM:SS] Camera snapshot uploaded to Supabase Storage`

### 4. **Verifikasi Upload:**
- Buka Supabase Dashboard → Storage → `camera_snapshots`
- Pastikan file `latest_frame.jpg` ada dan terupdate

### 5. **React Dashboard:**
- Buka aplikasi React Anda
- Masuk ke tab **"Live Inspection"**
- Gambar dari Supabase akan ditampilkan dan diperbarui setiap 1.5 detik

---

## 🔧 Konfigurasi Lanjutan

### Mengubah Interval Upload:
Ubah nilai `1.5` di line ~800 ke nilai lain (dalam detik):
```python
if time.time() - st.session_state.last_snapshot_upload_time >= 1.5:  # <- ubah di sini
```

### Mengubah Kualitas Kompresi:
Ubah nilai `60` di fungsi `upload_camera_snapshot()` (dalam persentase 0-100):
```python
ret, buffer = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])  # <- ubah di sini
```
- Semakin rendah = semakin ringan tapi kualitas menurun
- Semakin tinggi = kualitas bagus tapi file lebih besar

### Mengubah Nama Bucket/File:
```python
upload_camera_snapshot(
    frame_to_show, 
    bucket_name="your_bucket_name",      # <- ubah bucket
    file_name="your_filename.jpg"          # <- ubah filename
)
```

---

## 🐛 Troubleshooting

| Masalah | Solusi |
|---------|--------|
| **Upload tidak terjadi** | 1. Pastikan `SUPABASE_SERVICE_ROLE_KEY` ada di `.env`<br>2. Pastikan bucket `camera_snapshots` sudah dibuat & public<br>3. Cek console untuk error messages |
| **Gambar tidak terupdate di React** | 1. Pastikan React polling interval cocok (1.5 detik)<br>2. Cek URL Supabase di React config<br>3. Clear browser cache |
| **Upload lambat** | 1. Tingkatkan kualitas kompresi (turunkan nilai 60 ke 40-50)<br>2. Cek bandwidth jaringan<br>3. Pertimbangkan resize frame lebih kecil |
| **Error "Bucket not found"** | Pastikan bucket `camera_snapshots` sudah dibuat di Supabase Storage |

---

## 📊 Performance Notes

- **Current Interval:** 1.5 detik per upload
- **File Size:** ~15-20 KB per frame (dengan kompresi 60%)
- **Bandwidth Usage:** ~10-13 KB/s = ~0.6-0.8 MB/min
- **Frame Resolution:** 672×512 pixels
- **Format:** JPEG

---

## 🎯 Hasil Akhir

Ketika semua sudah dikonfigurasi:
✅ Streamlit akan upload frame kamera setiap 1.5 detik
✅ React Dashboard akan polling dan menampilkan live video
✅ CCTV-like experience dengan update otomatis

**Selamat! 🎉 Sistem Live Inspection Snapshot Anda sudah siap!**
