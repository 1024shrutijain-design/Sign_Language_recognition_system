import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import subprocess
 
# ── Load the New Topological Model ────────────────────────────
print("Loading Random Forest Model...")
try:
    with open('sign_lang_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("ERROR: sign_lang_model.pkl not found!")
    exit()
 
# ── Initialize MediaPipe ─────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
 
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
 
# ── Setup Text-to-Speech ─────────────────────────────────────
def speak(text):
    subprocess.Popen(['espeak', '-s', '140', '-v', 'en', text])
 
# ── Hardware Optimized Camera ────────────────────────────────
print("Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
 
# ── State & Variables ────────────────────────────────────────
word = ""
sentence = ""
CONF_THRESHOLD = 0.75       # 75% confidence required
SECONDS_TO_HOLD = 1.5       # Time required to lock in a letter
locked_letter = ""
hold_start_time = 0.0
last_added_letter = ""
 
fps_time = time.time()
 
# Resizable window for X11/VNC forwarding
cv2.namedWindow("SignBridge MTech", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SignBridge MTech", 1024, 768)
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    current_time = time.time()
 
    current_letter = ""
    current_conf = 0.0
 
    # ── MediaPipe Processing ─────────────────────────────────
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
 
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
       
        # Draw the Cyberpunk Skeleton on the hand
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
 
        # Extract 63-number Vector Math
        landmark_vector = []
        wrist = hand_landmarks.landmark[0]
        for lm in hand_landmarks.landmark:
            landmark_vector.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])
 
        # ── AI Prediction ────────────────────────────────────
        # Get probability scores for all classes
        probs = model.predict_proba([landmark_vector])[0]
        current_conf = max(probs)
       
        # Get the letter with the highest probability
        guessed_class_index = np.argmax(probs)
        current_letter = model.classes_[guessed_class_index]
 
    # ── Auto-Type Dwell Logic ────────────────────────────────
    is_valid_sign = (current_letter and
                     current_conf >= CONF_THRESHOLD and
                     current_letter.lower() not in ['nothing', 'space', 'del'])
 
    if is_valid_sign:
        if current_letter == locked_letter:
            time_held = current_time - hold_start_time
            if time_held >= SECONDS_TO_HOLD and current_letter != last_added_letter:
                word += current_letter
                last_added_letter = current_letter
                print(f"Auto-Added: {current_letter} | Word: {word}")
        else:
            locked_letter = current_letter
            hold_start_time = current_time
            last_added_letter = ""
    else:
        locked_letter = ""
        hold_start_time = current_time
 
    # ── Draw Advanced UI ─────────────────────────────────────
    # Top Panel Background
    cv2.rectangle(frame, (0, 0), (w, 130), (20, 20, 40), -1)
 
    # Big Detected Letter
    cv2.putText(frame, current_letter if current_letter else "—",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 100), 4)
 
    # Confidence Score & Bar
    cv2.putText(frame, f"ACC: {current_conf*100:.0f}%",
                (140, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 220, 0), 2)
    bar_fill = int(220 * current_conf)
    bar_color = (0, 200, 80) if current_conf > CONF_THRESHOLD else (0, 140, 220)
    cv2.rectangle(frame, (140, 65), (360, 85), (50, 50, 50), -1)
    cv2.rectangle(frame, (140, 65), (140 + bar_fill, 85), bar_color, -1)
 
    # Visual Dwell Timer (Draws near the bottom of the screen)
    if is_valid_sign and locked_letter == current_letter:
        time_held = current_time - hold_start_time
        if time_held < SECONDS_TO_HOLD and last_added_letter != current_letter:
            cv2.putText(frame, f"Locking: {time_held:.1f}s / {SECONDS_TO_HOLD}s",
                        (w//2 - 100, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            progress = int((time_held / SECONDS_TO_HOLD) * 300)
            cv2.line(frame, (w//2 - 150, h - 70), (w//2 - 150 + progress, h - 70), (0, 255, 0), 6)
        elif last_added_letter == current_letter:
            cv2.putText(frame, "ADDED!", (w//2 - 60, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
 
    # Bottom Panel Background
    cv2.rectangle(frame, (0, h - 60), (w, h), (20, 20, 40), -1)
 
    # Current Word Output
    cv2.putText(frame, f"BUFFER: {word}_",
                (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
 
    # Full Sentence Output
    cv2.putText(frame, f"OUT: {sentence[-45:]}",
                (300, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
 
    # Instructions
    cv2.putText(frame, "SPACE=add | ENTER=speak | BKSP=del | C=clear | Q=quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
 
    # FPS Counter
    fps = 1.0 / (time.time() - fps_time + 0.001)
    fps_time = time.time()
    cv2.putText(frame, f"Cam FPS: {fps:.0f}",
                (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 2)
 
    cv2.imshow("SignBridge MTech", frame)
 
    # ── Controls ─────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        if current_letter and current_conf >= CONF_THRESHOLD:
            word += current_letter
    elif key == 13:  
        if word:
            sentence += word + " "
            speak(word)
            word = ""
    elif key == 8:    
        word = word[:-1]
    elif key == ord('c'):
        word, sentence = "", ""
 
cap.release()
cv2.destroyAllWindows()