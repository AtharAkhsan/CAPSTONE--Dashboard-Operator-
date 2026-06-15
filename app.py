"""
==========================================================================
  CAPSTONE PROJECT — Dashboard Operator
  Sistem Verifikasi Kuantitas Part Mikro
  Sensor Fusion: Kamera AI (Density Map Estimation) + Load Cell
==========================================================================
  Framework  : Streamlit
  AI Model   : MobileNetV2 + Dilated Convolution (DME)
  Checkpoint : checkpoints/final_dme_97percent.pth
==========================================================================
"""

import os
import time
import random
import sqlite3
import uuid
import json           # <--- Baru
import serial       # <--- Baru
from datetime import datetime

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

# Import arsitektur model dari file proyek
from model_dme import DensityMapRegressor


# ============================================================
# KONSTANTA & KONFIGURASI
# ============================================================

CHECKPOINT_PATH = os.path.join("checkpoints", "best_dme_model.pth")

# Resolusi target model (sama seperti saat training)
TARGET_W, TARGET_H = 672, 512

# Resolusi asli webcam
CAM_W, CAM_H = 1920, 1080

# Ukuran center crop (4:3) untuk membuang distorsi lensa cembung di pinggir
CROP_W, CROP_H = 1440, 1080

# Normalisasi standar ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Scaling factor yang digunakan saat training loss
SCALING_FACTOR = 1000.0

# Database Master Parts
PARTS_DB = {
    "JPS-0001": {"name": "JP Screw", "vendor": "PT. Why-Fi", "target_qty": 100, "base_weight": 0.87},
    "SPR-0012": {"name": "Spur Gear 2.5g", "vendor": "PT. Sejahtera", "target_qty": 100, "base_weight": 2.50}
}


# ============================================================
# DATABASES: SQLite & Supabase
# ============================================================

load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

@st.cache_resource
def init_supabase(api_key):
    if SUPABASE_URL and api_key:
        try:
            return create_client(SUPABASE_URL, api_key)
        except Exception as e:
            st.error(f"Supabase init error: {e}")
    return None

supabase_client = init_supabase(SUPABASE_ANON_KEY)
supabase_write_client = init_supabase(SUPABASE_SERVICE_ROLE_KEY)

def init_sqlite():
    conn = sqlite3.connect("local_data.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS verification_logs (
            id TEXT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            part_code TEXT,
            target_qty REAL,
            ai_count REAL,
            load_cell_count REAL,
            final_count REAL,
            diff_pct REAL,
            image_url TEXT,
            status TEXT,
            is_synced INTEGER DEFAULT 0
        )
    ''')

    # Migrasi ringan untuk DB lama yang belum punya kolom image_url.
    cursor.execute("PRAGMA table_info(verification_logs)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if "image_url" not in existing_cols:
        cursor.execute("ALTER TABLE verification_logs ADD COLUMN image_url TEXT")

    conn.commit()
    return conn


def save_app_config(key, value):
    """Simpan konfigurasi aplikasi sederhana ke tabel `app_config`."""
    conn = st.session_state.sqlite_conn
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS app_config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    cursor.execute('REPLACE INTO app_config (key, value) VALUES (?, ?)', (key, json.dumps(value)))
    conn.commit()


def load_app_config(key, default=None):
    """Muat konfigurasi dari `app_config` jika ada, kembalikan default jika tidak."""
    conn = st.session_state.sqlite_conn
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT value FROM app_config WHERE key = ?', (key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    return default

if "sqlite_conn" not in st.session_state:
    st.session_state.sqlite_conn = init_sqlite()

def save_to_local_db(part_code, target_qty, ai_count, lc_count, final_count, diff_pct, status, image_url=None, log_id=None):
    if log_id is None:
        log_id = str(uuid.uuid4())

    conn = st.session_state.sqlite_conn
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO verification_logs (id, part_code, target_qty, ai_count, load_cell_count, final_count, diff_pct, image_url, status, is_synced)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
    ''', (log_id, part_code, target_qty, float(ai_count), float(lc_count), float(final_count), float(diff_pct), image_url, status))
    conn.commit()
    return log_id

def sync_to_supabase():
    if not supabase_write_client:
        return False, "SUPABASE_SERVICE_ROLE_KEY belum dikonfigurasi. RLS memblokir insert/update dengan anon key."
    
    conn = st.session_state.sqlite_conn
    cursor = conn.cursor()
    cursor.execute('SELECT id, timestamp, part_code, target_qty, ai_count, load_cell_count, final_count, diff_pct, image_url, status FROM verification_logs WHERE is_synced = 0')
    rows = cursor.fetchall()
    
    if not rows:
        return True, "Semua data sudah sinkron dengan Supabase."

    payloads = []
    synced_ids = []

    for row in rows:
        # Konversi SQLite CURRENT_TIMESTAMP (YYYY-MM-DD HH:MM:SS) ke format ISO8601
        ts = row[1]
        if isinstance(ts, str) and ' ' in ts:
            ts = ts.replace(' ', 'T') + 'Z'

        payloads.append({
            "id": row[0],
            "timestamp": ts,
            "part_code": row[2],
            "target_qty": row[3],
            "ai_count": row[4],
            "load_cell_count": row[5],
            "final_count": row[6],
            "diff_pct": row[7],
            "image_url": row[8],
            "status": row[9],
        })
        synced_ids.append(row[0])

    try:
        # Upsert menangani data baru dan data duplikat berdasarkan primary key `id`.
        supabase_write_client.table("verification_logs").upsert(payloads, on_conflict="id").execute()

        cursor.executemany(
            'UPDATE verification_logs SET is_synced = 1 WHERE id = ?',
            [(row_id,) for row_id in synced_ids],
        )
        conn.commit()

        return True, f"Berhasil sinkronisasi {len(synced_ids)} data."
    except Exception as e:
        print(f"Batch sync error: {e}")
        return False, f"Gagal sinkronisasi ke Supabase: {e}"


def sync_single_log_to_supabase(log_id):
    """
    Sinkronisasi satu log berdasarkan id.
    Dipakai saat manual save agar dashboard bisa langsung membaca log terbaru.
    """
    if not supabase_write_client:
        return False, "SUPABASE_SERVICE_ROLE_KEY belum dikonfigurasi."

    conn = st.session_state.sqlite_conn
    cursor = conn.cursor()
    cursor.execute(
        'SELECT id, timestamp, part_code, target_qty, ai_count, load_cell_count, final_count, diff_pct, image_url, status FROM verification_logs WHERE id = ?',
        (log_id,),
    )
    row = cursor.fetchone()
    if not row:
        return False, f"Log {log_id} tidak ditemukan di database lokal."

    ts = row[1]
    if isinstance(ts, str) and ' ' in ts:
        ts = ts.replace(' ', 'T') + 'Z'

    payload = {
        "id": row[0],
        "timestamp": ts,
        "part_code": row[2],
        "target_qty": row[3],
        "ai_count": row[4],
        "load_cell_count": row[5],
        "final_count": row[6],
        "diff_pct": row[7],
        "image_url": row[8],
        "status": row[9],
    }

    try:
        supabase_write_client.table("verification_logs").upsert(payload, on_conflict="id").execute()
        cursor.execute('UPDATE verification_logs SET is_synced = 1 WHERE id = ?', (log_id,))
        conn.commit()
        return True, "Log berhasil sinkron ke Supabase."
    except Exception as e:
        return False, f"Gagal sinkron log ke Supabase: {e}"


# ============================================================
# FUNGSI: Supabase Telemetry Storage (Table-Based)
# ============================================================
def save_telemetry_to_supabase(part_code, part_name, vendor, target_qty, ai_count, weight_data, base_weight, decision, status, discrepancy):
    """
    Simpan snapshot telemetry ke tabel Supabase.
    Ini adalah jalur aman untuk telemetry: tidak memakai broadcast channel.
    """
    if not supabase_write_client:
        return False, "SUPABASE_SERVICE_ROLE_KEY belum dikonfigurasi."

    payload = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "part_code": part_code,
        "part_name": part_name,
        "vendor": vendor,
        "target_qty": target_qty,
        "ai_count": ai_count,
        "weight_data": weight_data,
        "base_weight": base_weight,
        "decision": decision,
        "status": status,
        "discrepancy": discrepancy,
    }

    try:
        supabase_write_client.table("telemetry_logs").insert(payload).execute()
        return True, "Telemetry tersimpan ke Supabase."
    except Exception as e:
        return False, f"Gagal simpan telemetry ke Supabase: {e}"


# ============================================================
# FUNGSI: Device Selection
# ============================================================

def select_device():
    """Deteksi device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# FUNGSI: Load Model (cached agar tidak reload setiap frame)
# ============================================================

@st.cache_resource
def load_model():
    """
    Load model DensityMapRegressor dari checkpoint.
    Menggunakan @st.cache_resource agar model hanya di-load SEKALI
    dan tetap di memory selama aplikasi berjalan.
    """
    device = select_device()

    model = DensityMapRegressor(pretrained=False)
    model = model.to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch    = checkpoint.get("epoch", "?")
    best_mae = checkpoint.get("best_mae", "?")

    return model, device, epoch, best_mae


# ============================================================
# FUNGSI: Inference Transform
# ============================================================

@st.cache_resource
def get_inference_transform():
    """Pipeline preprocessing: Normalize + ToTensor (tanpa resize)."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ============================================================
# FUNGSI: Center Crop
# ============================================================

def center_crop(frame, crop_w=CROP_W, crop_h=CROP_H):
    """
    Potong bagian tengah frame untuk membuang distorsi lensa cembung.
    Input  : frame 1920x1080 (BGR)
    Output : frame 1440x1080 (BGR) — area tengah saja
    """
    h, w = frame.shape[:2]
    x_start = (w - crop_w) // 2
    y_start = (h - crop_h) // 2
    return frame[y_start : y_start + crop_h, x_start : x_start + crop_w]


# ============================================================
# FUNGSI: Pre-processing Frame untuk Model
# ============================================================

def preprocess_frame(frame_rgb, transform):
    """
    Resize frame ke TARGET (672x512) lalu normalize + convert ke tensor.
    Returns: (tensor, display_frame_rgb)
      - tensor         : untuk input ke model
      - display_frame  : frame 672x512 RGB untuk ditampilkan
    """
    resized = cv2.resize(frame_rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    display_frame = resized.copy()

    transformed = transform(image=resized)
    tensor = transformed["image"]  # (C, H, W)

    return tensor, display_frame


# ============================================================
# FUNGSI: Inference → Density Map + Count
# ============================================================

def run_inference(model, tensor, device):
    """
    Jalankan forward pass model pada satu frame.
    Returns: (density_map_numpy, predicted_count)
    """
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)  # (1, 1, H, W)

    # Hilangkan scaling factor training
    density_map = (output / SCALING_FACTOR).squeeze().cpu().numpy()
    predicted_count = float(density_map.sum())

    return density_map, predicted_count


# ============================================================
# FUNGSI: Heatmap Overlay
# ============================================================

def create_heatmap_overlay(display_frame_rgb, density_map, alpha=0.5):
    """
    Gabungkan density map heatmap (JET) dengan frame RGB.
    Keduanya sudah berukuran 672x512.
    """
    if density_map.max() > 0:
        norm = (density_map / density_map.max() * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(density_map, dtype=np.uint8)

    heatmap_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(display_frame_rgb, alpha, heatmap_rgb, 1 - alpha, 0)
    return overlay


# ============================================================
# FUNGSI: Upload Camera Snapshot ke Supabase Storage
# ============================================================

def upload_camera_snapshot(frame_rgb, bucket_name="camera_snapshots", file_name="latest_frame.jpg"):
    """
    Upload frame gambar ke Supabase Storage untuk Live Inspection di React Dashboard.
    
    Args:
        frame_rgb: Frame RGB dari OpenCV (NumPy array)
        bucket_name: Nama bucket di Supabase Storage (default: "camera_snapshots")
        file_name: Nama file di bucket (default: "latest_frame.jpg") — akan di-upsert
    
    Returns:
        True jika upload berhasil, False jika gagal
    """
    if not supabase_write_client:
        return False
    
    try:
        # Konversi RGB ke BGR untuk OpenCV (OpenCV bekerja dengan BGR)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Encode frame menjadi JPEG dengan kompresi 60% (lebih ringan & cepat upload)
        ret, buffer = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        
        if not ret:
            print("[Camera Snapshot] Gagal encode frame ke JPEG")
            return False
        
        # Konversi buffer ke byte array
        file_bytes = buffer.tobytes()
        
        # Upload ke Supabase Storage dengan upsert=True (menimpa file lama)
        res = supabase_write_client.storage.from_(bucket_name).upload(
            file_name,
            file_bytes,
            file_options={"content_type": "image/jpeg", "upsert": "true"}
        )
        
        return True
    
    except Exception as e:
        print(f"[Camera Snapshot] Error uploading snapshot: {e}")
        return False


def upload_inspection_proof(frame_rgb, log_id, bucket_name="camera_snapshots"):
    """
    Upload snapshot bukti inspeksi per-log dan kembalikan public URL-nya.
    File menggunakan nama unik berbasis log_id agar cocok dengan timestamp log.
    """
    if not supabase_write_client:
        return None

    try:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ret:
            print("[Inspection Proof] Gagal encode frame ke JPEG")
            return None

        file_name = f"snapshot_{log_id}.jpg"
        file_bytes = buffer.tobytes()

        supabase_write_client.storage.from_(bucket_name).upload(
            file_name,
            file_bytes,
            file_options={"content_type": "image/jpeg", "upsert": "true"},
        )

        public_url = supabase_write_client.storage.from_(bucket_name).get_public_url(file_name)
        if isinstance(public_url, dict):
            return public_url.get("publicURL") or public_url.get("publicUrl")
        return public_url

    except Exception as e:
        print(f"[Inspection Proof] Error upload proof: {e}")
        return None


# ============================================================
# FUNGSI: Pembacaan Load Cell via Serial (Real-Time)
# ============================================================

# testing

def get_load_cell_weight(ai_count, base_weight):
    """
    Membaca data JSON dari Arduino Uno via Serial.
    Menggunakan in_waiting agar tidak memblokir (lagging) frame rate kamera.
    """
    # Inisialisasi nilai konfigurasi/calibration di session state
    if "last_raw_weight" not in st.session_state:
        st.session_state.last_raw_weight = 0.0
    if "last_weight" not in st.session_state:
        st.session_state.last_weight = 0.0
    if "lc_calibration_factor" not in st.session_state:
        # muat dari DB jika tersedia
        st.session_state.lc_calibration_factor = load_app_config("lc_calibration_factor", 1.0)
    if "lc_tare_raw" not in st.session_state:
        st.session_state.lc_tare_raw = load_app_config("lc_tare_raw", 0.0)

    if "serial_conn" in st.session_state and st.session_state.serial_conn is not None:
        try:
            ser = st.session_state.serial_conn
            last_valid_line = None

            # Kuras semua data yang menumpuk di buffer, ambil yang paling baru
            while ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                # Pastikan format JSON utuh atau format teks dari Arduino
                if line.startswith("{") and line.endswith("}"):
                    last_valid_line = line
                elif "berat terbaca:" in line:
                    last_valid_line = line

            if last_valid_line:
                if last_valid_line.startswith("{"):
                    data = json.loads(last_valid_line)
                    # Ambil key "berat" sesuai format print JSON dari Arduino
                    raw = float(data.get("berat", 0.0))
                    st.session_state.last_raw_weight = raw
                elif "berat terbaca:" in last_valid_line:
                    # Format: berat terbaca: 10.5 g | berat satuan: 0.0 g | jumlah barang: 0 pcs
                    parts = last_valid_line.split("|")
                    if len(parts) > 0:
                        berat_str = parts[0].replace("berat terbaca:", "").replace("g", "").strip()
                        st.session_state.last_raw_weight = float(berat_str)

        except Exception:
            # Jika ada error parsing (misal kabel tersenggol), biarkan pakai nilai terakhir
            pass

    # Konversi raw -> gram menggunakan faktor kalibrasi dan offset tare
    grams = (st.session_state.last_raw_weight - st.session_state.lc_tare_raw) * st.session_state.lc_calibration_factor
    # Lindungi dari -ve kecil
    if grams < 0 and abs(grams) < 0.0001:
        grams = 0.0

    st.session_state.last_weight = float(grams)
    return st.session_state.last_weight


# ============================================================
# FUNGSI: Logika Sensor Fusion
# ============================================================

def sensor_fusion(ai_count, live_weight, base_weight, target_qty):
    """
    Logika Sensor Fusion:
    """
    weight_estimation_pcs = round(live_weight / base_weight)
    
    # Menghitung nilai rata-rata dari estimasi berat dan estimasi kamera vision
    final_count = round((weight_estimation_pcs + round(ai_count)) / 2)

    # Keputusan final harus dibandingkan terhadap target, menggunakan final count
    discrepancy = final_count - target_qty
    reference_count = max(abs(target_qty), 1)
    discrepancy_pct = abs(discrepancy) / reference_count * 100

    if discrepancy_pct <= 3:
        status = "PASS"
        status_color = "green"
        text = "OK"
    else:
        status = "REJECT"
        status_color = "red"
        text = "NG"

    return weight_estimation_pcs, final_count, discrepancy, status, status_color, text


# ============================================================
# STREAMLIT: Page Config & Custom CSS
# ============================================================

st.set_page_config(
    page_title="Dashboard Operator — Verifikasi Part Mikro",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS untuk tampilan industrial / profesional
st.markdown("""
<style>
    /* --- Font --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* --- Header bar --- */
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 18px 28px;
        border-radius: 12px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 900;
        margin: 0;
        letter-spacing: 1px;
    }
    .main-header .subtitle {
        color: #8ecae6;
        font-size: 0.85rem;
        font-weight: 400;
    }

    /* --- Panel card --- */
    .panel-card {
        background: #1a1a2e;
        border: 1px solid #30305a;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .panel-card h3 {
        color: #8ecae6;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* --- Status badges --- */
    .status-verified {
        background: linear-gradient(90deg, #064e3b, #059669);
        color: #d1fae5;
        padding: 14px 20px;
        border-radius: 10px;
        font-size: 1.15rem;
        font-weight: 700;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 1px;
    }
    .status-warning {
        background: linear-gradient(90deg, #7f1d1d, #991b1b);
        color: #fca5a5;
        padding: 14px 20px;
        border-radius: 10px;
        font-size: 1.15rem;
        font-weight: 700;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 1px;
    }

    /* --- Metric override --- */
    [data-testid="stMetric"] {
        background: #16213e;
        border: 1px solid #30305a;
        border-radius: 10px;
        padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8ecae6 !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 900;
        font-size: 2rem;
        color: #ffffff !important;
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background: #0d1b2a;
    }

    /* --- Hide Streamlit branding --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# STREAMLIT: Header
# ============================================================

st.markdown("""
<div class="main-header">
    <div>
        <h1>🏭 Dashboard Operator — Verifikasi Part Mikro</h1>
        <span class="subtitle">Sensor Fusion: Kamera AI (Density Map Estimation) + Load Cell</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# STREAMLIT: Sidebar — Kontrol Kamera & Info Model
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Kontrol Sistem")
    st.divider()
    
    # Pilihan Part
    st.markdown("### 📦 Informasi Part")
    selected_part_code = st.selectbox("Pilih Part Code", options=list(PARTS_DB.keys()))
    part_info = PARTS_DB[selected_part_code]
    
    st.markdown(f"**Name:** {part_info['name']}")
    st.markdown(f"**Vendor:** {part_info['vendor']}")
    st.markdown(f"**Target Qty:** {part_info['target_qty']} pcs")
    st.markdown(f"**Base Weight:** {part_info['base_weight']} g")
    
    st.divider()

    # Tombol Start / Stop
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶ START", width="stretch", type="primary"):
            st.session_state.camera_running = True
    with col_stop:
        if st.button("⏹ STOP", width="stretch"):
            st.session_state.camera_running = False

    status_text = "🟢 AKTIF" if st.session_state.camera_running else "🔴 NONAKTIF"
    st.markdown(f"**Status Kamera:** {status_text}")

    st.divider()

    # Pengaturan kamera
    st.markdown("### 📷 Pengaturan Kamera")
    camera_index = st.selectbox("Indeks Kamera", [0, 1, 2], index=0)
    show_heatmap = st.checkbox("Tampilkan Heatmap Overlay", value=True)
    heatmap_alpha = st.slider("Opacity Frame (vs Heatmap)", 0.2, 0.8, 0.5, 0.05)
    # Telemetry aman: simpan snapshot ke tabel Supabase
    enable_telemetry = st.checkbox("Simpan Telemetry ke Supabase", value=False)
    if enable_telemetry and not SUPABASE_SERVICE_ROLE_KEY:
        st.warning("Aktifkan `SUPABASE_SERVICE_ROLE_KEY` di `.env` untuk menyimpan telemetry ke Supabase.")

    st.divider()

    # Load Cell

    st.markdown("### 🔌 Koneksi Timbangan (Arduino)")
    
    # Deteksi OS untuk memberikan contoh format port yang relevan
    contoh_port = "COM3" if os.name == 'nt' else "/dev/ttyUSB0"
    com_port = st.text_input("Port Serial", value=contoh_port)
    
    col_conn, col_disc = st.columns(2)
    with col_conn:
        if st.button("🔌 Connect", width="stretch"):
            try:
                # Tutup koneksi lama jika ada
                if "serial_conn" in st.session_state and st.session_state.serial_conn is not None:
                    st.session_state.serial_conn.close()
                
                # Buka port dengan baudrate 9600 dan timeout sangat kecil agar tidak lag
                st.session_state.serial_conn = serial.Serial(com_port, 9600, timeout=0.05)
                time.sleep(2) # Tunggu Arduino reset sesaat setelah port dibuka
                st.toast(f"✅ Terhubung ke {com_port}", icon="✅")
            except Exception as e:
                st.error(f"Gagal koneksi: Periksa port dan kabel!")
                st.session_state.serial_conn = None

    with col_disc:
        if st.button("Disconnect", width="stretch"):
            if "serial_conn" in st.session_state and st.session_state.serial_conn is not None:
                st.session_state.serial_conn.close()
                st.session_state.serial_conn = None
                st.toast("🔌 Serial diputus.")

    # Status indikator
    if "serial_conn" in st.session_state and st.session_state.serial_conn is not None and st.session_state.serial_conn.is_open:
        st.markdown("**Status Load Cell:** 🟢 Terhubung")
    else:
        st.markdown("**Status Load Cell:** 🔴 Terputus")

    # --- Kalibrasi & Tare ---
    if "lc_calibration_factor" not in st.session_state:
        st.session_state.lc_calibration_factor = load_app_config("lc_calibration_factor", 1.0)
    if "lc_tare_raw" not in st.session_state:
        st.session_state.lc_tare_raw = load_app_config("lc_tare_raw", 0.0)
    if "last_raw_weight" not in st.session_state:
        st.session_state.last_raw_weight = 0.0
    if "last_weight" not in st.session_state:
        st.session_state.last_weight = 0.0

    st.markdown("### ⚖️ Kalibrasi Timbangan")
    st.markdown(f"- Faktor Kalibrasi: **{st.session_state.lc_calibration_factor:.6f}**")
    st.markdown(f"- Offset Tare (raw): **{st.session_state.lc_tare_raw:.3f}**")
    st.markdown(f"- Last Raw: **{st.session_state.last_raw_weight:.3f}** → {st.session_state.last_weight:.2f} g")

    col_tare, col_cal = st.columns(2)
    with col_tare:
        if st.button("TARE (Set Zero)", use_container_width=True):
            # Ambil pembacaan raw terakhir sebagai tare
            st.session_state.lc_tare_raw = float(st.session_state.get("last_raw_weight", 0.0))
            save_app_config("lc_tare_raw", st.session_state.lc_tare_raw)
            st.toast(f"Tare diset ke {st.session_state.lc_tare_raw:.3f} (raw)")
    with col_cal:
        if st.button("Reset Kalibrasi", use_container_width=True):
            st.session_state.lc_calibration_factor = 1.0
            save_app_config("lc_calibration_factor", st.session_state.lc_calibration_factor)
            st.toast("Faktor kalibrasi direset ke 1.0")

    st.markdown("#### Kalibrasi dengan beban diketahui")
    known_w = st.number_input("Known weight (grams)", min_value=0.0, value=50.0, step=1.0)
    if st.button("Kalibrasi dengan Known Weight", use_container_width=True):
        measured_raw = float(st.session_state.get("last_raw_weight", 0.0))
        tare = float(st.session_state.get("lc_tare_raw", 0.0))
        denom = (measured_raw - tare)
        if denom <= 0:
            st.error("Tidak dapat kalibrasi: pembacaan raw <= tare. Pastikan beban ada dan stabil.")
        else:
            factor = float(known_w) / denom
            st.session_state.lc_calibration_factor = factor
            save_app_config("lc_calibration_factor", st.session_state.lc_calibration_factor)
            st.toast(f"Kalibrasi selesai: faktor={st.session_state.lc_calibration_factor:.6f}")

    st.markdown("#### Atur manual faktor kalibrasi")
    new_factor = st.number_input("Manual factor", min_value=0.000001, value=float(st.session_state.lc_calibration_factor), format="%.6f")
    if st.button("Simpan Faktor Manual", use_container_width=True):
        st.session_state.lc_calibration_factor = float(new_factor)
        save_app_config("lc_calibration_factor", st.session_state.lc_calibration_factor)
        st.toast(f"Faktor kalibrasi disimpan: {st.session_state.lc_calibration_factor:.6f}")

    st.divider()
    
    # Info model
    st.markdown("### 🧠 Info Model AI")
    if os.path.exists(CHECKPOINT_PATH):
        model, device, epoch, best_mae = load_model()
        st.success(f"Model loaded pada **{device}**")
        st.markdown(f"- **Arsitektur:** MobileNetV2 + Dilated Conv")
        st.markdown(f"- **Checkpoint:** `final_dme_97percent.pth`")
        st.markdown(f"- **Epoch:** {epoch}")
        st.markdown(f"- **Best MAE:** {best_mae}")
        st.markdown(f"- **Input Size:** {TARGET_W}×{TARGET_H}")
    else:
        st.error(f"Checkpoint tidak ditemukan!\n`{CHECKPOINT_PATH}`")
        st.stop()

    st.divider()
    st.markdown("### ☁️ Sinkronisasi Data")
    if SUPABASE_SERVICE_ROLE_KEY:
        if st.button("🔄 Sync ke Supabase", width="stretch"):
            with st.spinner("Menyinkronkan data..."):
                success, msg = sync_to_supabase()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
    else:
        st.button("🔄 Sync ke Supabase", width="stretch", disabled=True)
        st.warning(
            "SUPABASE_SERVICE_ROLE_KEY belum dikonfigurasi. Tambahkan ke `.env` agar sinkronisasi bisa menulis ke tabel yang dilindungi RLS."
        )
        st.code(
            "SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here",
            language="text",
        )
                
    st.divider()
    st.caption("Capstone Project — Sistem Verifikasi Kuantitas Part Mikro")


# ============================================================
# STREAMLIT: Layout Utama — 2 Kolom
# ============================================================

col_camera, col_panel = st.columns([3, 2], gap="large")

# --- Kolom Kiri: Live Camera Feed ---
with col_camera:
    st.markdown('<div class="panel-card"><h3>📹 Live Camera Feed</h3></div>',
                unsafe_allow_html=True)
    frame_placeholder = st.empty()
    fps_placeholder   = st.empty()
    
    st.markdown("---")
    st.markdown("### 💾 Simpan Data")
    if "manual_save_requested" not in st.session_state:
        st.session_state.manual_save_requested = False

    if st.button("💾 Save ke Database Lokal", width="stretch", disabled=not st.session_state.camera_running):
        st.session_state.manual_save_requested = True
        st.toast("Permintaan simpan diterima. Data akan disimpan dari pembacaan terbaru.")

    if not st.session_state.camera_running:
        st.caption("Aktifkan kamera (START) untuk mengaktifkan tombol simpan.")
    else:
        st.caption("Klik untuk menyimpan satu record dari pembacaan terbaru.")

# --- Kolom Kanan: Panel Indikator ---
with col_panel:
    st.markdown('<div class="panel-card"><h3>📊 Panel Verifikasi</h3></div>',
                unsafe_allow_html=True)
                
    st.markdown(f"**Current Inspection:** {part_info['name']} ({selected_part_code})")
    
    st.markdown("---")

    metric_ai     = st.empty()
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        metric_lc     = st.empty()
    with col_w2:
        metric_lc_count = st.empty()
    metric_final  = st.empty()

    st.markdown("---")
    status_placeholder = st.empty()
    diff_placeholder   = st.empty()

    st.markdown("---")
    log_placeholder = st.empty()


# ============================================================
# STREAMLIT: Loop Utama — Real-time Inference
# ============================================================

if st.session_state.camera_running:
    if "last_telemetry_time" not in st.session_state:
        st.session_state.last_telemetry_time = 0.0
    if "last_snapshot_upload_time" not in st.session_state:
        st.session_state.last_snapshot_upload_time = 0.0
        
    transform = get_inference_transform()
    cap = cv2.VideoCapture(camera_index)

    # Set resolusi webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        st.error("❌ Gagal membuka kamera! Periksa koneksi dan indeks kamera.")
        st.session_state.camera_running = False
        st.stop()

    frame_count = 0

    while st.session_state.camera_running:
        t_start = time.time()

        ret, frame_bgr = cap.read()
        if not ret:
            st.warning("⚠️ Tidak dapat membaca frame dari kamera.")
            break

        # --------------------------------------------------
        # STEP 1: Center Crop (1920x1080 → 1440x1080)
        # Membuang distorsi cembung di pinggir lensa
        # --------------------------------------------------
        cropped = center_crop(frame_bgr, CROP_W, CROP_H)

        # Konversi ke RGB
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # --------------------------------------------------
        # STEP 2: Resize ke 672x512 + Normalisasi → Tensor
        # --------------------------------------------------
        tensor, display_frame = preprocess_frame(cropped_rgb, transform)

        # --------------------------------------------------
        # STEP 3: Inference AI — Density Map Estimation
        # --------------------------------------------------
        density_map, ai_count = run_inference(model, tensor, device)

        # --------------------------------------------------
        # STEP 4: Heatmap Overlay
        # --------------------------------------------------
        if show_heatmap:
            overlay = create_heatmap_overlay(display_frame, density_map, alpha=heatmap_alpha)
            frame_to_show = overlay
        else:
            frame_to_show = display_frame

        # --------------------------------------------------
        # STEP 5: Load Cell -> GRAM
        # --------------------------------------------------
        live_weight_g = get_load_cell_weight(ai_count, part_info['base_weight'])

        # --------------------------------------------------
        # STEP 6: Sensor Fusion
        # --------------------------------------------------
        weight_est_pcs, final_count, discrepancy, status, status_color, text_ng = sensor_fusion(ai_count, live_weight_g, part_info['base_weight'], part_info['target_qty'])

        # --------------------------------------------------
        # STEP 7: Update UI
        # --------------------------------------------------
        frame_placeholder.image(frame_to_show, caption="Live Feed (672×512)", width="stretch")

        # Metrics
        metric_ai.metric("🤖 AI Visual Count", f"{round(ai_count)} pcs")
        metric_lc.metric("⚖️ Live Weight Data", f"{live_weight_g:.2f} g")
        metric_lc_count.metric("⚖️ Weight Count", f"{weight_est_pcs} pcs")
        metric_final.metric("📊 Final Count", f"{final_count} pcs")

        # Status badge
        if status_color == "green":
            status_placeholder.markdown(
                f'<div class="status-verified">Final Decision: {status}</div>',
                unsafe_allow_html=True,
            )
        else:
            status_placeholder.markdown(
                f'<div class="status-warning">Final Decision: {status}</div>',
                unsafe_allow_html=True,
            )

        diff_placeholder.markdown(
            f"<p style='text-align:center; color:#94a3b8; font-size:0.9rem;'>"
            f"Discrepancy: <b>{discrepancy} pcs, {text_ng}</b></p>",
            unsafe_allow_html=True,
        )

        # FPS
        t_end = time.time()
        fps = 1.0 / max(t_end - t_start, 1e-6)
        frame_count += 1
        fps_placeholder.caption(f"⚡ FPS: {fps:.1f} — Frame #{frame_count}")

        # Log
        log_placeholder.markdown(
            f"<div style='background:#0d1b2a; padding:10px; border-radius:8px;"
            f" font-family:JetBrains Mono,monospace; font-size:0.75rem; color:#64748b;'>"
            f"[{time.strftime('%H:%M:%S')}] "
            f"AI={round(ai_count)} | W={live_weight_g:.1f}g | "
            f"Final={final_count} | Δ={discrepancy} | "
            f"Status={status}</div>",
            unsafe_allow_html=True,
        )

        # --------------------------------------------------
        # STEP 8: Upload Camera Snapshot ke Supabase (Setiap 1.5 detik)
        # --------------------------------------------------
        if SUPABASE_SERVICE_ROLE_KEY and (time.time() - st.session_state.last_snapshot_upload_time >= 1.5):
            upload_ok = upload_camera_snapshot(frame_to_show, bucket_name="camera_snapshots", file_name="latest_frame.jpg")
            if upload_ok:
                print(f"[{time.strftime('%H:%M:%S')}] Camera snapshot uploaded to Supabase Storage")
            st.session_state.last_snapshot_upload_time = time.time()

        # --------------------------------------------------
        # STEP 9: Manual Save ke Local DB (via tombol)
        # --------------------------------------------------
        if st.session_state.get("manual_save_requested", False):
            log_id = str(uuid.uuid4())
            proof_url = None

            if SUPABASE_SERVICE_ROLE_KEY:
                proof_url = upload_inspection_proof(frame_to_show, log_id=log_id, bucket_name="camera_snapshots")

            save_to_local_db(
                selected_part_code,
                part_info['target_qty'],
                round(ai_count),
                live_weight_g,
                final_count,
                discrepancy,
                status,
                image_url=proof_url,
                log_id=log_id,
            )

            if SUPABASE_SERVICE_ROLE_KEY:
                sync_ok, sync_msg = sync_single_log_to_supabase(log_id)
                if not sync_ok:
                    st.toast(sync_msg)

            st.session_state.manual_save_requested = False
            if proof_url:
                st.toast("💾 Data & snapshot bukti inspeksi berhasil disimpan")
            else:
                st.toast("💾 Data tersimpan, tetapi URL snapshot bukti belum tersedia")

        # --------------------------------------------------
        # STEP 10: Telemetry ke Supabase (Setiap ~1 detik jika diaktifkan)
        # --------------------------------------------------
        if enable_telemetry and SUPABASE_SERVICE_ROLE_KEY and (time.time() - st.session_state.last_telemetry_time >= 1.0):
            telemetry_ok, telemetry_msg = save_telemetry_to_supabase(
                part_code=selected_part_code,
                part_name=part_info['name'],
                vendor=part_info['vendor'],
                target_qty=part_info['target_qty'],
                ai_count=round(ai_count),
                weight_data=live_weight_g,
                base_weight=part_info['base_weight'],
                decision=status,
                status=("ok" if status == "PASS" else "ng"),
                discrepancy=discrepancy,
            )
            st.session_state.last_telemetry_time = time.time()
            if not telemetry_ok:
                st.toast(telemetry_msg)

    # Cleanup
    cap.release()

else:
    # Tampilan idle ketika kamera belum dinyalakan
    with col_camera:
        frame_placeholder.markdown(
            "<div style='background:#0d1b2a; border:2px dashed #30305a;"
            " border-radius:12px; height:400px; display:flex;"
            " align-items:center; justify-content:center;'>"
            "<p style='color:#64748b; font-size:1.1rem;'>"
            "📷 Tekan <b>START</b> di sidebar untuk memulai kamera"
            "</p></div>",
            unsafe_allow_html=True,
        )

    with col_panel:
        st.markdown(f"**Current Inspection:** {part_info['name']} ({selected_part_code})")
        st.markdown("---")
        metric_ai.metric("🤖 AI Visual Count", "—")
        metric_lc.metric("⚖️ Live Weight Data", "—")
        metric_lc_count.metric("⚖️ Weight Count", "—")
        metric_final.metric("📊 Final Count", "—")
        status_placeholder.markdown(
            '<div class="status-verified" style="opacity:0.4;">STANDBY</div>',
            unsafe_allow_html=True,
        )
