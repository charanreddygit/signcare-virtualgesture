import cv2
import mediapipe as mp
import numpy as np
import os

gesture = input("Enter gesture name: ")

path = f"dataset_g/{gesture}"
os.makedirs(path, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)

count = 0


while True:

    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    landmarks = []
    key = cv2.waitKey(1)

    if key == ord('s'):
        np.save(f"{path}/{count}.npy", landmarks)
        count += 1

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:
            for lm in hand.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

        while len(landmarks) < 126:
            landmarks.append(0)

        np.save(f"{path}/{count}.npy", landmarks)

        count += 1

    cv2.putText(frame,f"Samples: {count}",(10,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Collecting",frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # ESC key
        break
    

    

cap.release()
cv2.destroyAllWindows()