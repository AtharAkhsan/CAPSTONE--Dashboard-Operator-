# CAPSTONE PROJECT — Dashboard Operator

## Sistem Verifikasi Kuantitas Part Mikro

This project is a dashboard for operators to verify the quantity of micro parts using a combination of an AI Camera (Density Map Estimation) and a Load Cell (Sensor Fusion).

### Features
- **Framework**: Streamlit
- **AI Model**: MobileNetV2 + Dilated Convolution (DME)
- **Sensor Fusion**: Integrates AI density map estimation with physical weight from a load cell.
- **Database**: SQLite (Local) and Supabase (Cloud Sync)
- **Camera Integration**: Live preview with Heatmap Overlay.
- **Serial Communication**: Real-time reading from Arduino for the load cell.

### Setup and Installation

1. **Activate Virtual Environment** (Windows PowerShell):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   *(Note: If you encounter an execution policy error, you might need to run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first)*

2. **Install Dependencies**:
   If not already installed, make sure to install the required packages (e.g., using `pip install -r requirements.txt`).

3. **Environment Variables**:
   Create a `.env` file in the root directory and configure your Supabase credentials:
   ```env
   SUPABASE_URL=your_supabase_url
   SUPABASE_ANON_KEY=your_anon_key
   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

### Checkpoints
Make sure your model weights are located at `checkpoints/final_dme_97percent.pth`.
