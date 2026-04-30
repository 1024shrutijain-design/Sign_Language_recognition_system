import cv2
import mediapipe as mp
import csv
import numpy as np

# ── Initialize MediaPipe ─────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ── Create CSV File & Write Header ───────────────────────────
csv_file = 'dataset.csv'
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    # Header: x0, y0, z0, x1, y1, z1 ... label
    header = []
    for i in range(21):
        header.extend([f'x{i}', f'y{i}', f'z{i}'])
    header.append('label')
    writer.writerow(header)

print("Data Collector Started!")
print("Make a sign, then press that letter on your keyboard to record it.")
print("Press 'ESC' to quit.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    landmark_vector = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # ── Extact and Normalize Data ────────────────────────
            wrist = hand_landmarks.landmark[0]
            
            for lm in hand_landmarks.landmark:
                # Subtract wrist position to make all coordinates relative to the wrist
                landmark_vector.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])
                
    # UI Instructions
    cv2.putText(frame, "Press a letter key to record.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press ESC to quit.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Data Collection', frame)

    # ── Keyboard Logic ───────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27: # ESC key to break
        break
        
    # If a valid letter key is pressed AND a hand is on screen
    elif 97 <= key <= 122: 
        letter = chr(key).upper()
        if len(landmark_vector) == 63:
            # Append the letter label to the end of the 63 numbers
            landmark_vector.append(letter)
            
            # Write to CSV
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(landmark_vector)
                
            print(f"Recorded: {letter} | Total points saved: {len(landmark_vector)-1}")
        else:
            print(f"Cannot record '{letter}' - No hand detected!")

cap.release()
cv2.destroyAllWindows()
print("Data collection complete. Check dataset.csv")