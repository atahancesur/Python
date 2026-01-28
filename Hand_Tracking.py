import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time

# ==============================
#  AYARLAR
# ==============================
MAX_HANDS = 1          # Aynı anda 1 el üzerinden kontrol
SHOW_LANDMARK_IDS = False  # Noktaların numarasını yazmak istersen True yap
DRAW_THICKNESS = 6     # Air drawing çizgi kalınlığı
SMOOTHING = 0.2        # Mouse hareketi yumuşatma katsayısı (0-1 arası)

# Ekran boyutu (mouse kontrol için)
SCREEN_W, SCREEN_H = pyautogui.size()

# Kamera
cap = cv2.VideoCapture(0)

# Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Air drawing için canvas
canvas = None

# Mouse pozisyonu için önceki değer (yumuşatma)
prev_mouse_x, prev_mouse_y = 0, 0

# Tıklama debouncing
last_click_time = 0
CLICK_COOLDOWN = 0.25  # saniye


# ==============================
#  YARDIMCI FONKSİYONLAR
# ==============================

def get_hand_landmarks(image, hand_landmarks):
    """Landmark'ları piksel koordinatlarına çevirir."""
    h, w, _ = image.shape
    points = []
    for lm in hand_landmarks.landmark:
        cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
        points.append((cx, cy, cz))
    return points


def fingers_up(lm_list):
    """
    Hangi parmakların havada olduğunu basitçe tespit eder.
    lm_list: 21 noktadan oluşan liste (x, y, z)
    Dönüş: [thumb, index, middle, ring, pinky] (1=havada, 0=kapalı)
    """
    fingers = [0, 0, 0, 0, 0]
    if not lm_list:
        return fingers

    # Başparmak: x karşılaştırması (sağ el varsayıyoruz, kamera flipped!)
    # 4: başparmak ucu, 3: eklem
    if lm_list[4][0] > lm_list[3][0]:
        fingers[0] = 1

    # Diğer 4 parmak için: tip (8,12,16,20) PIP'den (6,10,14,18) yukarıda mı?
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for i in range(1, 5):
        tip_y = lm_list[finger_tips[i-1]][1]
        pip_y = lm_list[finger_pips[i-1]][1]
        if tip_y < pip_y:  # ekran koordinatında yukarı daha küçük y
            fingers[i] = 1

    return fingers


def distance(p1, p2):
    """İki nokta arası Öklid uzaklığı."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def recognize_gesture(lm_list, fingers_state):
    """
    Basit gesture tanıyıcı.
    Dönüş: string (gesture adı)
    """
    if not lm_list:
        return "None"

    total_up = sum(fingers_state)

    # Fist: tüm parmaklar kapalı
    if total_up == 0:
        return "Fist"

    # Open Hand: 4 veya 5 parmak açık
    if total_up >= 4:
        return "Open Hand"

    # Point: sadece işaret parmağı açık
    if fingers_state[1] == 1 and total_up == 1:
        return "Point"

    # OK işareti: başparmak ve işaret parmağı uçları birbirine yakın
    thumb_tip = lm_list[4]
    index_tip = lm_list[8]
    d = distance(thumb_tip, index_tip)
    if d < 40:  # piksel eşiği (kamera çözünürlüğüne göre değişebilir)
        return "OK"

    return "Unknown"


def control_mouse(lm_list, fingers_state):
    """
    İşaret parmağı ile mouse hareketi,
    başparmak + işaret parmağı pinch ile sol tıklama.
    """
    global prev_mouse_x, prev_mouse_y, last_click_time

    if not lm_list:
        return

    index_tip = lm_list[8]
    thumb_tip = lm_list[4]

    # İşaret parmağı açık ise imleci kontrol et
    if fingers_state[1] == 1:
        x, y = index_tip[0], index_tip[1]

        # Kameradaki koordinatı ekran koordinatına ölçekle
        frame_w, frame_h = 640, 480  # Klasik webcam boyutu varsayımı
        mouse_x = np.interp(x, (0, frame_w), (0, SCREEN_W))
        mouse_y = np.interp(y, (0, frame_h), (0, SCREEN_H))

        # Yumuşatma
        smooth_x = prev_mouse_x + (mouse_x - prev_mouse_x) * SMOOTHING
        smooth_y = prev_mouse_y + (mouse_y - prev_mouse_y) * SMOOTHING

        pyautogui.moveTo(smooth_x, smooth_y)
        prev_mouse_x, prev_mouse_y = smooth_x, smooth_y

    # Başparmak + işaret parmağı pinch ise tıklama
    d = distance((thumb_tip[0], thumb_tip[1]), (index_tip[0], index_tip[1]))
    current_time = time.time()
    if d < 35 and (current_time - last_click_time) > CLICK_COOLDOWN:
        pyautogui.click()
        last_click_time = current_time


def update_canvas(canvas, lm_list, fingers_state):
    """
    Air drawing:
    - İşaret parmağı açık, orta parmak kapalıysa ÇİZİM modunda.
    """
    if canvas is None or not lm_list:
        return canvas

    index_tip = lm_list[8]
    middle_tip = lm_list[12]

    drawing_mode = fingers_state[1] == 1 and fingers_state[2] == 0

    if drawing_mode:
        x, y = index_tip[0], index_tip[1]
        cv2.circle(canvas, (x, y), DRAW_THICKNESS // 2, (255, 255, 255), -1)

    return canvas


# ==============================
#  ANA DÖNGÜ
# ==============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Aynaya benzemesi için flip
    frame = cv2.flip(frame, 1)

    # Canvas ilk frame'de oluşturuluyor
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    lm_list = []
    fingers_state = [0, 0, 0, 0, 0]
    gesture_name = "None"
    depth_text = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # El iskeletini çiz
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Landmark'ları al
            lm_list = get_hand_landmarks(frame, hand_landmarks)

            # İstenirse her noktayı numarasını yazarak göster
            if SHOW_LANDMARK_IDS:
                for idx, (x, y, z) in enumerate(lm_list):
                    cv2.putText(frame, str(idx), (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Parmaklar
            fingers_state = fingers_up(lm_list)
            finger_count = sum(fingers_state)

            # Gesture
            gesture_name = recognize_gesture(lm_list, fingers_state)

            # Mouse kontrolü
            control_mouse(lm_list, fingers_state)

            # Air drawing
            canvas = update_canvas(canvas, lm_list, fingers_state)

            # Z derinliğine göre "Yakın/Uzak" (0: bilek)
            wrist_z = lm_list[0][2]
            if wrist_z < -0.1:
                depth_text = "El Kameraya Yakin"
            else:
                depth_text = "El Kameradan Uzak"

            break  # MAX_HANDS=1 olduğu için tek el yeterli

    # Canvas'ı görüntü üzerine bind et
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray_canvas, 5, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    # UI yazıları
    cv2.rectangle(frame, (0, 0), (350, 120), (0, 0, 0), -1)

    cv2.putText(frame, f"Fingers: {sum(fingers_state)}  {fingers_state}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Gesture: {gesture_name}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, depth_text,
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(frame, "Q: Cikis | C: Temizle",
                (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Hand All-in-One", frame)

    key = cv2.waitKey(1) & 0xFF

    # C -> Canvas temizle
    if key == ord('c'):
        canvas = np.zeros_like(frame)

    # Q -> Cikis
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
