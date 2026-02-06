import cv2
import math
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import mediapipe as mp

VOL_MIN_DIST, VOL_MAX_DIST = 20, 150          
BAR_TOP, BAR_BOT = 150, 400                   

mp_hands = mp.solutions.hands
hands     = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw  = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
if devices is None:
    raise RuntimeError("No speaker device found—pycaw failed.")
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume    = interface.QueryInterface(IAudioEndpointVolume)
MIN_VOL, MAX_VOL, _ = volume.GetVolumeRange()     # e.g. (‑65.25, 0.0, 0.03125)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try a different index.")


try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)

        lm = []
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]          
            h, w, _ = frame.shape
            for idx, pt in enumerate(hand.landmark):
                lm.append((idx, int(pt.x * w), int(pt.y * h)))
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        
        if len(lm) >= 9:
            x1, y1 = lm[4][1:]
            x2, y2 = lm[8][1:]
            dist   = math.hypot(x2 - x1, y2 - y1)

            vol = np.interp(dist, [VOL_MIN_DIST, VOL_MAX_DIST], [MIN_VOL, MAX_VOL])
            volume.SetMasterVolumeLevel(vol, None)

            bar_y = int(np.interp(dist, [VOL_MIN_DIST, VOL_MAX_DIST], [BAR_BOT, BAR_TOP]))
            pct   = int(np.interp(dist, [VOL_MIN_DIST, VOL_MAX_DIST], [0, 100]))

            cv2.rectangle(frame, (50, BAR_TOP), (85, BAR_BOT), (0, 255, 0), 2)
            cv2.rectangle(frame, (50, bar_y),   (85, BAR_BOT), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f"{pct} %", (40, BAR_BOT + 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Volume Control", frame)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
