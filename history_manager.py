# --- START OF FILE history_manager.py ---

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
import shutil
import uuid
import base64
import zipfile
import io

class HistoryManager:
    """
    Kelas untuk mengelola riwayat analisis forensik video secara komprehensif.
    Menyimpan data, artefak, dan menyediakan fungsi untuk antarmuka pengguna yang detail.
    """
    
    def __init__(self, history_file="analysis_history.json", history_folder="analysis_artifacts"):
        """
        Inisialisasi History Manager.
        
        Args:
            history_file (str): Nama file JSON untuk menyimpan data riwayat.
            history_folder (str): Folder untuk menyimpan semua artefak visual (plot, gambar).
        """
        self.history_file = Path(history_file)
        self.history_folder = Path(history_folder)
        self.history_folder.mkdir(exist_ok=True)

        # ====== [NEW] False-Positive Fix June-2025 ======
        self.db_path = Path("analysis_settings.db")
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS settings (id TEXT PRIMARY KEY, video_name TEXT, timestamp TEXT, fps_awal REAL, fps_baru REAL, ssim_thresh REAL, z_thresh REAL)"
        )
        conn.commit()
        conn.close()
        # ====== [END NEW] ======
        
        # Buat file riwayat jika belum ada dengan struktur list kosong.
        if not self.history_file.exists():
            with open(self.history_file, 'w') as f:
                json.dump([], f)
    
    def save_analysis(self, result, video_name, additional_info=None):
        """
        Menyimpan hasil analisis lengkap ke dalam file riwayat.
        
        Args:
            result: Objek AnalysisResult dari ForensikVideo.
            video_name (str): Nama file video yang dianalisis.
            additional_info (dict): Informasi tambahan opsional seperti FPS.
            
        Returns:
            str: ID unik dari entri riwayat yang baru saja disimpan.
        """
        analysis_id = str(uuid.uuid4())
        
        # Buat sub-folder spesifik untuk artefak analisis ini.
        artifact_folder = self.history_folder / analysis_id
        artifact_folder.mkdir(exist_ok=True)
        
        # Salin artefak visual penting ke folder riwayat.
        saved_artifacts = self._save_artifacts(result, artifact_folder)
        
        # ======================= FIX START =======================
        # Struktur data riwayat yang akan disimpan ke JSON.
        # Menggunakan 'forensic_evidence_matrix' yang baru, bukan 'integrity_analysis' yang lama.
        history_entry = {
            "id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "video_name": video_name,
            "artifacts_folder": str(artifact_folder),
            "preservation_hash": result.preservation_hash,
            "summary": result.summary,
            "metadata": result.metadata,
            "forensic_evidence_matrix": result.forensic_evidence_matrix if hasattr(result, 'forensic_evidence_matrix') else None,
            "localization_details": result.localization_details if hasattr(result, 'localization_details') else None,
            "pipeline_assessment": result.pipeline_assessment if hasattr(result, 'pipeline_assessment') else None,
            "localizations": result.localizations,
            "localizations_count": len(result.localizations),
            "anomaly_types": self._count_anomaly_types(result),
            "saved_artifacts": saved_artifacts,
            "additional_info": additional_info if additional_info else {},
            "report_paths": { # Hanya menyimpan path laporan Markdown
                "markdown": str(result.markdown_report_path) if hasattr(result, 'markdown_report_path') and result.markdown_report_path else None,
            },
        }
        # ======================= FIX END =======================
        
        history = self.load_history()
        history.append(history_entry)
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=4)

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO settings(id, video_name, timestamp, fps_awal, fps_baru, ssim_thresh, z_thresh) VALUES (?,?,?,?,?,?,?)",
                (
                    analysis_id,
                    video_name,
                    history_entry["timestamp"],
                    additional_info.get("fps_awal") if additional_info else None,
                    additional_info.get("fps_baru") if additional_info else None,
                    additional_info.get("ssim_threshold") if additional_info else None,
                    additional_info.get("z_threshold") if additional_info else None,
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

        return analysis_id
    
    def load_history(self):
        """
        Memuat seluruh riwayat analisis dari file JSON.
        
        Returns:
            list: Daftar semua entri riwayat analisis.
        """
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            with open(self.history_file, 'w') as f:
                json.dump([], f)
            return []
    
    def get_analysis(self, analysis_id):
        """
        Mengambil satu entri riwayat berdasarkan ID uniknya.
        
        Args:
            analysis_id (str): ID analisis yang dicari.
            
        Returns:
            dict: Entri riwayat yang ditemukan, atau None jika tidak ada.
        """
        history = self.load_history()
        return next((entry for entry in history if entry["id"] == analysis_id), None)
    
    def delete_analysis(self, analysis_id):
        """
        Menghapus satu entri riwayat dan semua artefak terkait.
        
        Args:
            analysis_id (str): ID analisis yang akan dihapus.
            
        Returns:
            bool: True jika berhasil dihapus, False jika tidak.
        """
        history = self.load_history()
        
        entry_to_delete = self.get_analysis(analysis_id)
        if not entry_to_delete:
            return False

        artifact_folder = Path(entry_to_delete.get("artifacts_folder", ""))
        if artifact_folder.exists() and artifact_folder.is_dir():
            shutil.rmtree(artifact_folder)
        
        updated_history = [entry for entry in history if entry["id"] != analysis_id]
        
        with open(self.history_file, 'w') as f:
            json.dump(updated_history, f, indent=4)
                
        return True
    
    def delete_all_history(self):
        """
        Menghapus SEMUA riwayat analisis dan semua artefaknya. Operasi ini tidak dapat diurungkan.
        
        Returns:
            int: Jumlah entri yang berhasil dihapus.
        """
        history = self.load_history()
        count = len(history)
        
        if self.history_folder.exists():
            shutil.rmtree(self.history_folder)
        self.history_folder.mkdir(exist_ok=True)
        
        with open(self.history_file, 'w') as f:
            json.dump([], f)

        return count

    def _format_timestamp(self, iso_timestamp):
        """
        Memformat timestamp ISO menjadi format yang lebih mudah dibaca.
        """
        try:
            dt = datetime.fromisoformat(iso_timestamp)
            return dt.strftime("%d %B %Y, %H:%M:%S")
        except (ValueError, TypeError):
            return iso_timestamp    
    def export_analysis(self, analysis_id):
        """
        Mengekspor data analisis lengkap (metadata + artefak) sebagai file ZIP.
        
        Args:
            analysis_id (str): ID analisis yang akan diekspor.
            
        Returns:
            bytes: Data file ZIP dalam bentuk bytes, atau None jika gagal.
        """
        entry = self.get_analysis(analysis_id)
        if not entry:
            return None
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Tambahkan laporan Markdown ke ZIP
            report_path_str = entry.get("report_paths", {}).get("markdown")
            if report_path_str and Path(report_path_str).exists():
                report_path = Path(report_path_str)
                zip_file.write(report_path, arcname=report_path.name)

            # 2. Tambahkan semua artefak visual ke dalam folder 'artifacts' di dalam ZIP
            saved_artifacts = entry.get("saved_artifacts", {})
            for key, artifact_path_str in saved_artifacts.items():
                if "report" not in key: # Jangan sertakan laporan lagi
                    artifact_path = Path(artifact_path_str)
                    if artifact_path.exists() and artifact_path.is_file():
                        zip_file.write(artifact_path, arcname=f"artifacts/{artifact_path.name}")
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    def _count_anomaly_types(self, result):
        """Helper untuk menghitung jumlah setiap jenis anomali."""
        counts = {"duplication": 0, "insertion": 0, "discontinuity": 0}
        for loc in getattr(result, 'localizations', []):
            event_type = loc.get('event', '').replace('anomaly_', '')
            if event_type in counts:
                counts[event_type] += 1
        return counts
    
    def _save_artifacts(self, result, folder):
        """Helper untuk menyalin artefak visual penting ke folder riwayat."""
        saved = {}
        
        for plot_name, plot_path in result.plots.items():
            if isinstance(plot_path, (str, Path)) and os.path.exists(plot_path):
                target_path = folder / Path(plot_path).name
                shutil.copy(plot_path, target_path)
                saved[plot_name] = str(target_path)
        
        localizations = getattr(result, 'localizations', [])
        for i, loc in enumerate(localizations[:3]): # Simpan hanya 3 contoh anomali pertama
            if loc.get('image') and os.path.exists(loc['image']):
                target_path = folder / f"sample_anomaly_frame_{i}.jpg"
                shutil.copy(loc['image'], target_path)
                saved[f"anomaly_frame_{i}"] = str(target_path) # Kunci menjadi 'anomaly_frame_0', 'anomaly_frame_1', dll

        if getattr(result, 'markdown_report_path', None) and os.path.exists(result.markdown_report_path):
            target_path = folder / Path(result.markdown_report_path).name
            shutil.copy(result.markdown_report_path, target_path)
            saved['markdown_report'] = str(target_path)
        
        return saved
    
    def get_artifact_base64(self, artifact_path):
        """Mengonversi file gambar artefak menjadi string base64 untuk ditampilkan di web."""
        path = Path(artifact_path)
        if not path.is_file():
            return None
            
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            mime_type = "image/png" if path.suffix.lower() == '.png' else "image/jpeg"
            return f"data:{mime_type};base64,{data}"
        except Exception:
            return None

    def get_anomaly_description(self, anomaly_type):
        """Menyediakan deskripsi lengkap untuk setiap jenis anomali."""
        descriptions = {
            "duplication": {
                "title": "Duplikasi Frame", "icon": "🔁", "color": "#FF6B6B",
                "simple": "Frame yang sama diulang beberapa kali dalam video.",
                "technical": "Dideteksi melalui perbandingan pHash dan dikonfirmasi dengan SIFT+RANSAC yang menemukan kecocokan fitur yang sangat tinggi antar frame.",
                "implication": "Ini bisa menjadi indikasi untuk memperpanjang durasi secara artifisial atau untuk menyembunyikan/menutupi konten yang telah dihapus di antara frame yang diduplikasi.",
                "example": "Seperti Anda menyalin sebuah halaman dari buku dan menempelkannya lagi di tempat lain untuk membuat buku terlihat lebih tebal."
            },
            "discontinuity": {
                "title": "Diskontinuitas Video", "icon": "✂️", "color": "#45B7D1",
                "simple": "Terjadi 'lompatan' atau patahan mendadak dalam aliran visual atau gerakan video.",
                "technical": "Dideteksi melalui penurunan drastis pada skor SSIM (kemiripan struktural) atau lonjakan tajam pada magnitudo Optical Flow (aliran gerakan).",
                "implication": "Seringkali ini adalah tanda kuat dari pemotongan (cut) dan penyambungan (paste) video. Aliran alami video terganggu.",
                "example": "Bayangkan sebuah kalimat di mana beberapa kata di tengahnya hilang, membuat kalimatnya terasa aneh dan melompat."
            },
            "insertion": {
                "title": "Penyisipan Konten", "icon": "➕", "color": "#4ECDC4",
                "simple": "Adanya frame atau segmen baru yang tidak ada di video asli/baseline.",
                "technical": "Dideteksi secara definitif dengan membandingkan hash setiap frame dari video bukti dengan video baseline. Frame yang ada di bukti tapi tidak di baseline dianggap sebagai sisipan.",
                "implication": "Ini adalah bukti kuat dari penambahan konten yang bisa mengubah konteks atau narasi video secara signifikan.",
                "example": "Seperti menambahkan sebuah paragraf karangan Anda sendiri ke tengah-tengah novel karya orang lain."
            }
        }
        return descriptions.get(anomaly_type, {
            "title": "Anomali Lain", "icon": "❓", "color": "#808080", "simple": "Jenis anomali tidak dikenali.",
            "technical": "-", "implication": "-", "example": "-"
        })

    # ======================= FIX START =======================
    # Menghapus fungsi get_integrity_explanation yang sudah usang
    # def get_integrity_explanation(self, score): ...
    # ======================= FIX END =======================