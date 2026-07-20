import cv2
import mediapipe as mp

# ==========================================
# 1. INISIALISASI MEDIAPIPE & KONFIGURASI
# ==========================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Konfigurasi Efek
BLUR_KERNEL = (61, 61)   # Harus berupa angka ganjil
HOLD_FRAMES = 5          # Jumlah frame buffer agar efek blur tidak berkedip (flicker)

# ==========================================
# 2. FUNGSI DETEKSI GESTUR
# ==========================================
def is_finger_up(tip_idx, pip_idx, landmarks):
    """Memeriksa apakah ujung jari berada di atas sendinya (posisi berdiri)."""
    return landmarks[tip_idx].y < landmarks[pip_idx].y

def is_peace_sign(landmarks):
    """Deteksi gestur Peace (Telunjuk & Tengah berdiri, sisanya ditekuk)."""
    # Telunjuk (8) dan Tengah (12) HARUS berdiri
    index_up = is_finger_up(8, 6, landmarks)
    middle_up = is_finger_up(12, 10, landmarks)

    # Manis (16) dan Kelingking (20) HARUS ditekuk
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y

    # Jempol (4) ditekuk: Ujung jempol berada di bawah pangkal jari tengah (9)
    # Cara ini jauh lebih stabil dibanding mengecek koordinat X
    thumb_folded = landmarks[4].y > landmarks[9].y

    return index_up and middle_up and ring_down and pinky_down and thumb_folded

# ==========================================
# 3. MAIN LOOP (KAMERA)
# ==========================================
cap = cv2.VideoCapture(0)
blur_counter = 0  # Counter untuk menjaga efek blur tetap halus

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Gagal mengakses kamera.")
        break

    # Cermin & ubah ke RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Proses deteksi tangan
    results = hands.process(rgb_frame)
    detected_in_this_frame = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark tangan di layar
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Cek gestur peace
            if is_peace_sign(hand_landmarks.landmark):
                detected_in_this_frame = True
                break

    # Kelola buffer frame untuk menghindari efek blur kedap-kedip
    if detected_in_this_frame:
        blur_counter = HOLD_FRAMES
    elif blur_counter > 0:
        blur_counter -= 1

    # Terapkan Efek Blur jika gestur terdeteksi
    if blur_counter > 0:
        frame = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)
        cv2.putText(
            frame, 
            "PEACE DETECTED: BLUR ACTIVE", 
            (20, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 0), 
            2
        )

    # Tampilkan output
    cv2.imshow("Peace Gesture Blur Effect", frame)

    # Tekan 'ESC' untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
