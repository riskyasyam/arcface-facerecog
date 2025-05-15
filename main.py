import cv2
import joblib
import numpy as np
from insightface.app import FaceAnalysis

# Load model yang sudah disimpan
clf = joblib.load('notebooks/model_face_recognizer.pkl')
le = joblib.load('notebooks/label_encoder.pkl')

# Inisialisasi FaceAnalysis dari insightface
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Buka kamera
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("❌ Gagal membuka kamera.")
    exit()

print("✅ Kamera dibuka. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Tidak bisa membaca frame.")
        break

    # Deteksi wajah
    faces = app.get(frame)

    for face in faces:
        bbox = face.bbox.astype(int)
        emb = face.embedding.reshape(1, -1)

        # Prediksi nama dengan model SVM
        pred = clf.predict(emb)
        prob = clf.predict_proba(emb).max()
        name = le.inverse_transform(pred)[0]

        # Gambar kotak dan label
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        label = f'{name} ({prob:.2f})'
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow('Face Recognition - InsightFace + SVM', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()