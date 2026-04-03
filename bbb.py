import cv2 as cv
import mediapipe as mp
import math
import win32api
import win32con
import win32gui

# -------- NATIVE WINDOWS VOLUME USING win32api ----------
def set_volume(level):  # level: 0.0 to 1.0
    vol_value = int(level * 100)
    
    WM_APPCOMMAND = 0x0319
    APPCOMMAND_VOLUME_UP = 10
    win32gui.SendMessage(win32gui.GetForegroundWindow(),
                         WM_APPCOMMAND,
                         None,
                         APPCOMMAND_VOLUME_UP * 0x10000)

    # Set exact level using keyboard events
    for _ in range(50):  # bring to 0
        win32api.keybd_event(174, 0, 0, 0)  # volume down

    # raise to desired level
    for _ in range(vol_value):
        win32api.keybd_event(175, 0, 0, 0)  # volume up
# --------------------------------------------------------


vid = cv.VideoCapture(0)
vid.set(3, 640)
vid.set(4, 420)

mp_hands = mp.solutions.hands.Hands()
drawing = mp.solutions.drawing_utils

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = vid.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)
    hand_landmarks = result.multi_hand_landmarks

    x1 = y1 = x2 = y2 = None

    if hand_landmarks:
        for hand in hand_landmarks:
            drawing.draw_landmarks(frame, hand)
            for id, lm in enumerate(hand.landmark):

                h, w, c = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)

                if id == 8:
                    cv.circle(frame, (x, y), 6, (0, 255, 0), -1)
                    x1, y1 = x, y
                if id == 4:
                    cv.circle(frame, (x, y), 6, (0, 255, 0), -1)
                    x2, y2 = x, y

                if x1 is not None and x2 is not None:
                    cv.line(frame, (x1, y1), (x2, y2), (0,255,255), 3)

                    dist = math.hypot(x2 - x1, y2 - y1)
                    vol = min(max(dist / 200, 0), 15)

                    set_volume(vol)

                    cv.putText(frame, f"Volume: {int(vol*100)}%", (20, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (15,255,20), 2)

    cv.imshow("Face + Hand Detection", frame)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()
