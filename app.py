import cv2
import numpy as np
import os
import requests
import time
from datetime import datetime
from insightface.app import FaceAnalysis
from flask import Flask, Response
from dotenv import load_dotenv
import threading
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. KONFIGURASI DAN INISIALISASI ---

# Muat variabel dari file .env
load_dotenv()
DEVICE_ID = os.getenv('DEVICE_ID')
LARAVEL_BASE_URL = os.getenv('LARAVEL_BASE_URL')

# Buat URL lengkap untuk API Laravel
LARAVEL_STUDENTS_API_URL = f"{LARAVEL_BASE_URL}/api/devices/{DEVICE_ID}/students"
LARAVEL_ATTENDANCE_API_URL = f"{LARAVEL_BASE_URL}/api/absensi-siswa"

# Validasi konfigurasi .env
if not DEVICE_ID or not LARAVEL_BASE_URL:
    raise Exception("KRITIS: Pastikan DEVICE_ID dan LARAVEL_BASE_URL sudah diatur di file .env")

# Inisialisasi aplikasi Flask dan model InsightFace
app = Flask(__name__)
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# Buka koneksi ke kamera
cap = cv2.VideoCapture(0)

# Variabel Global untuk video stream dan galeri wajah
output_frame = None
lock = threading.Lock()
known_face_gallery = []  # Format: [{'id': id_siswa, 'embedding': embedding}]

print("="*50)
print(f"✅ Inisialisasi Device ID: {DEVICE_ID}")
print(f"✅ URL Get Siswa: {LARAVEL_STUDENTS_API_URL}")
print(f"✅ URL Post Absensi: {LARAVEL_ATTENDANCE_API_URL}")
print("="*50)

# --- 2. LOGIKA GALERI DINAMIS ---

def load_face_gallery():
    """
    Melakukan GET request ke Laravel untuk mengambil daftar siswa dan foto,
    lalu membangun galeri embedding wajah di memori.
    """
    global known_face_gallery
    print("🔄 Memuat galeri wajah dari server Laravel...")
    try:
        # Panggil API Laravel untuk mendapatkan data siswa
        response = requests.get(LARAVEL_STUDENTS_API_URL, timeout=15)
        if response.status_code != 200:
            print(f"❌ Gagal mengambil data siswa. Status: {response.status_code}, Pesan: {response.text}")
            return

        students_data = response.json().get('data', [])
        temp_gallery = []

        if not students_data:
            print("⚠️ Tidak ada data siswa yang diterima dari server untuk device ini.")
            return

        for student in students_data:
            student_id = student.get('id')
            # Loop melalui setiap foto yang dimiliki siswa
            for photo in student.get('fotos', []):
                photo_url = photo.get('url')
                if not photo_url:
                    continue

                # Download foto dari URL
                try:
                    img_response = requests.get(photo_url, stream=True, timeout=10)
                    if img_response.status_code == 200:
                        # Ubah data gambar menjadi format yang bisa dibaca OpenCV
                        img_array = np.asarray(bytearray(img_response.content), dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        
                        # Buat embedding dari foto
                        faces = face_app.get(img)
                        if faces:
                            embedding = faces[0].embedding
                            temp_gallery.append({'id': student_id, 'embedding': embedding})
                            print(f"✅ Foto untuk siswa ID: {student_id} berhasil diproses.")
                            break # Pindah ke siswa berikutnya setelah 1 foto berhasil diproses
                    else:
                        print(f"⚠️ Gagal download foto untuk siswa ID: {student_id} (Status: {img_response.status_code})")
                except Exception as e:
                    print(f"❌ Error saat memproses foto siswa ID {student_id}: {e}")
        
        # Ganti galeri lama dengan yang baru
        known_face_gallery = temp_gallery
        print(f"✅ Galeri wajah berhasil dimuat. Total {len(known_face_gallery)} wajah siswa dikenali.")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error koneksi saat memuat galeri: {e}")

# --- 3. LOGIKA DETEKSI DAN PEMBANDINGAN ---

def recognize_and_compare():
    """Fungsi ini berjalan di background, membandingkan wajah dengan galeri dinamis."""
    global output_frame, lock
    
    last_seen = {}
    COOLDOWN_SECONDS = 30 # Jeda 30 detik untuk siswa yang sama

    while True:
        if not cap.isOpened() or not known_face_gallery:
            time.sleep(5)
            # Coba muat ulang galeri jika masih kosong
            if not known_face_gallery:
                load_face_gallery()
            continue

        ret, frame = cap.read()
        if not ret:
            continue
        
        faces = face_app.get(frame)
        current_time = time.time()

        for face in faces:
            bbox = face.bbox.astype(int)
            live_embedding = face.embedding.reshape(1, -1)
            best_match_id = None
            best_similarity = 0.0

            # Bandingkan wajah yang terdeteksi dengan setiap wajah di galeri
            for known_face in known_face_gallery:
                similarity = cosine_similarity(live_embedding, known_face['embedding'].reshape(1, -1))[0][0]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = known_face['id']
            
            # Threshold kemiripan (sesuaikan nilai 0.5 jika perlu)
            if best_similarity > 0.5:
                label = f'ID: {best_match_id} ({best_similarity:.2f})'
                color = (0, 255, 0) # Hijau
                
                # Cek cooldown sebelum mengirim data
                if best_match_id not in last_seen or (current_time - last_seen[best_match_id]) > COOLDOWN_SECONDS:
                    print(f"✅ Cocok: Siswa ID {best_match_id}. Mengirim data absensi...")
                    threading.Thread(target=send_attendance, args=(best_match_id,)).start()
                    last_seen[best_match_id] = current_time
            else:
                label = 'Tidak Dikenal'
                color = (0, 0, 255) # Merah
            
            # Gambar kotak dan label di frame
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        with lock:
            output_frame = frame.copy()

# === FUNGSI YANG DIPERBARUI ===
def send_attendance(student_id):
    """Mengirim data absensi ke API Laravel dan mencetak respons server."""
    try:
        payload = {
            'id_siswa': int(student_id),
            'devices_id': int(DEVICE_ID),
            'waktu': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'status': 'hadir',
        }
        
        print(f"📤 Mengirim payload: {payload}") # Log payload yang dikirim
        
        response = requests.post(LARAVEL_ATTENDANCE_API_URL, json=payload, timeout=10)
        
        # Selalu cetak status dan respons dari server untuk debugging
        print(f"📬 Respons diterima dari server:")
        print(f"   - Status Code: {response.status_code}")
        print(f"   - Body: {response.text}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR KONEKSI saat mengirim absensi: {e}")
    except Exception as e:
        print(f"❌ ERROR LAIN saat mengirim absensi: {e}")

# --- 4. FLASK API ROUTES & MAIN EXECUTION ---

def generate_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    return f"<h1>API Presensi Dinamis Aktif</h1><p>Device ID: <strong>{DEVICE_ID}</strong></p><p>Total Wajah di Galeri: {len(known_face_gallery)}</p>"

@app.route("/video_feed")
def video_feed():
    """Endpoint untuk live video stream."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    # Saat aplikasi pertama kali dimulai, langsung muat galeri wajah
    load_face_gallery()

    # Jalankan fungsi deteksi di thread terpisah agar tidak memblokir server Flask
    recognition_thread = threading.Thread(target=recognize_and_compare)
    recognition_thread.daemon = True
    recognition_thread.start()
    
    # Jalankan aplikasi web Flask
    app.run(host='0.0.0.0', port=5000, debug=False)

# Cleanup
cap.release()
