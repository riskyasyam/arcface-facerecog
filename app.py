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
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# === KONFIGURASI & INISIALISASI ===

load_dotenv()
DEVICE_ID = os.getenv('DEVICE_ID')
LARAVEL_BASE_URL = os.getenv('LARAVEL_BASE_URL')

if not DEVICE_ID or not LARAVEL_BASE_URL:
    raise Exception("KRITIS: DEVICE_ID dan LARAVEL_BASE_URL wajib diatur di .env")

LARAVEL_STUDENTS_API_URL = f"{LARAVEL_BASE_URL}/api/devices/{DEVICE_ID}/students"
LARAVEL_ATTENDANCE_API_URL = f"{LARAVEL_BASE_URL}/api/absensi-siswa"

app = Flask(__name__)
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

cap = cv2.VideoCapture(0)
output_frame = None
lock = threading.Lock()
known_face_gallery = []

clf = None
le = None

print("=" * 60)
print(f"✅ DEVICE ID      : {DEVICE_ID}")
print(f"✅ GET SISWA API  : {LARAVEL_STUDENTS_API_URL}")
print(f"✅ POST ABSENSI   : {LARAVEL_ATTENDANCE_API_URL}")
print("=" * 60)

# === LOAD GALERI & LATIH SVM ===

def load_face_gallery():
    global known_face_gallery, clf, le
    print("🔄 Memuat galeri wajah dari server Laravel...")

    try:
        response = requests.get(LARAVEL_STUDENTS_API_URL, timeout=15)
        if response.status_code != 200:
            print(f"❌ Gagal mengambil data siswa. Status: {response.status_code}")
            return

        students_data = response.json().get('data', [])
        embeddings, labels = [], []
        temp_gallery = []

        for student in students_data:
            student_id = student.get('id')
            for photo in student.get('fotos', []):
                photo_url = photo.get('url')
                if not photo_url:
                    continue

                try:
                    img_response = requests.get(photo_url, stream=True, timeout=10)
                    if img_response.status_code == 200:
                        img_array = np.asarray(bytearray(img_response.content), dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        faces = face_app.get(img)
                        if faces:
                            for f in faces:
                                embedding = f.embedding
                                embeddings.append(embedding)
                                labels.append(str(student_id))
                                temp_gallery.append({'id': student_id, 'embedding': embedding})
                            print(f"✅ Siswa ID {student_id} - {len(faces)} embedding berhasil diproses.")
                    else:
                        print(f"⚠️ Gagal download foto siswa ID {student_id}")
                except Exception as e:
                    print(f"❌ Error proses foto siswa {student_id}: {e}")

        known_face_gallery = temp_gallery

        if embeddings and labels:
            if len(set(labels)) < 2:
                print("⚠️ Gagal melatih model: hanya ditemukan 1 kelas. Tambahkan lebih banyak siswa.")
                known_face_gallery = temp_gallery
                return

            le = LabelEncoder()
            y = le.fit_transform(labels)
            clf = SVC(probability=True)
            clf.fit(embeddings, y)
            print(f"✅ Model SVC dilatih dengan {len(embeddings)} embedding.")
        else:
            print("⚠️ Tidak ada embedding yang berhasil dikumpulkan.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error koneksi ke server Laravel: {e}")

# === DETEKSI DAN REKOGNISI WAJAH ===

def recognize_and_compare():
    global output_frame, lock
    last_seen = {}
    COOLDOWN_SECONDS = 30

    while True:
        if not cap.isOpened() or not known_face_gallery or clf is None or le is None:
            time.sleep(5)
            load_face_gallery()
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        faces = face_app.get(frame)
        current_time = time.time()

        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding.reshape(1, -1)

            pred = clf.predict(embedding)
            prob = clf.predict_proba(embedding).max()
            label = le.inverse_transform(pred)[0]

            if prob > 0.6:
                color = (0, 255, 0)
                label_text = f'{label} ({prob:.2f})'

                if label not in last_seen or (current_time - last_seen[label]) > COOLDOWN_SECONDS:
                    print(f"✅ Siswa ID {label} dikenali. Kirim absensi.")
                    threading.Thread(target=send_attendance, args=(label,)).start()
                    last_seen[label] = current_time
            else:
                color = (0, 0, 255)
                label_text = 'Tidak dikenal'

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label_text, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        with lock:
            output_frame = frame.copy()

# === KIRIM ABSENSI ===

def send_attendance(student_id):
    try:
        payload = {
            'id_siswa': int(student_id),
            'devices_id': int(DEVICE_ID),
            'waktu': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'status': 'hadir'
        }
        print(f"📤 Kirim absensi: {payload}")
        response = requests.post(LARAVEL_ATTENDANCE_API_URL, json=payload, timeout=10)
        print(f"📬 Status: {response.status_code}, Body: {response.text}")
    except Exception as e:
        print(f"❌ Gagal kirim absensi: {e}")

# === STREAMING FLASK ROUTE ===

def generate_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    return f"<h1>Presensi AI Aktif</h1><p>Device ID: {DEVICE_ID}</p><p>Galeri: {len(known_face_gallery)} wajah</p>"

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/ip")
def get_ip():
    return os.popen('hostname -I').read().split()[0]

# === MAIN ===

if __name__ == '__main__':
    load_face_gallery()
    threading.Thread(target=recognize_and_compare, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
    cap.release()