import cv2 as cv
import mediapipe as mp
import math
import time
import pyautogui

# pip install pyautogui

vid = cv.VideoCapture(0)
vid.set(3, 1280)
vid.set(4, 720)

mp_hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
drawing = mp.solutions.drawing_utils

# ---------- state ----------
zoom_level = 1.0
DIST_MIN, DIST_MAX = 30, 250
ZOOM_MIN,  ZOOM_MAX = 0.5, 4.0

# Zoom step cooldown (so it doesn't spam)
last_zoom_time = 0
ZOOM_COOLDOWN = 0.4          # seconds between zoom steps
last_zoom_dir = None         # "in" or "out"

# Middle-finger gesture for tab switching
MIDDLE_CLOSE_THRESH = 0.06
last_page_change_time = 0
PAGE_COOLDOWN = 1.2

# EMA for smooth distance reading
ema_dist = 140.0
EMA_ALPHA = 0.15

# Zoom bands — distance thresholds
ZOOM_IN_THRESHOLD  = 180     # fingers far apart → zoom in
ZOOM_OUT_THRESHOLD = 80      # fingers close → zoom out
NEUTRAL_BAND = (80, 180)     # inside this = no action

# ---------- helpers ----------

def landmark_dist(lm, a, b, W, H):
    ax, ay = lm[a].x * W, lm[a].y * H
    bx, by = lm[b].x * W, lm[b].y * H
    return math.hypot(bx - ax, by - ay)

def normalised_dist(lm, a, b):
    return math.hypot(lm[b].x - lm[a].x, lm[b].y - lm[a].y)

def is_middle_closed(lm):
    return normalised_dist(lm, 12, 9) < MIDDLE_CLOSE_THRESH

def zoom_chrome(direction):
    """Send Ctrl+= (zoom in) or Ctrl+- (zoom out) to whatever window is active."""
    if direction == "in":
        pyautogui.hotkey("ctrl", "=")
    elif direction == "out":
        pyautogui.hotkey("ctrl", "-")

def switch_tab(direction):
    """Ctrl+Tab = next tab, Ctrl+Shift+Tab = previous tab."""
    if direction == "next":
        pyautogui.hotkey("ctrl", "tab")
    else:
        pyautogui.hotkey("ctrl", "shift", "tab")

def draw_ruler(frame, dist):
    H, W = frame.shape[:2]
    bar_x, bar_y, bar_w, bar_h = 30, H - 50, W - 60, 14
    cv.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                 (50, 50, 50), -1)
    filled = int(((dist - DIST_MIN) / (DIST_MAX - DIST_MIN)) * bar_w)
    filled = min(max(filled, 0), bar_w)

    # Colour the bar based on zone
    if dist > ZOOM_IN_THRESHOLD:
        bar_color = (0, 255, 80)      # green = zooming in
    elif dist < ZOOM_OUT_THRESHOLD:
        bar_color = (0, 80, 255)      # red = zooming out
    else:
        bar_color = (0, 220, 180)     # teal = neutral

    cv.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                 bar_color, -1)

    cv.putText(frame, f"Dist: {int(dist)}px", (bar_x, bar_y - 8),
               cv.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv.putText(frame, f"[Neutral: {ZOOM_OUT_THRESHOLD}-{ZOOM_IN_THRESHOLD}px]",
               (bar_x + 160, bar_y - 8),
               cv.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

# ---------- main loop ----------
status_msg = ""
status_timer = 0

while True:
    ret, frame = vid.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    H, W = frame.shape[:2]

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)

    now = time.time()

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        # ── Smooth distance ──────────────────────────────────────────
        raw_dist = landmark_dist(lm, 4, 8, W, H)
        ema_dist = EMA_ALPHA * raw_dist + (1 - EMA_ALPHA) * ema_dist

        # Draw thumb/index indicators
        x1, y1 = int(lm[8].x * W), int(lm[8].y * H)
        x2, y2 = int(lm[4].x * W), int(lm[4].y * H)
        cv.circle(frame, (x1, y1), 8, (0, 255, 120), -1)
        cv.circle(frame, (x2, y2), 8, (0, 255, 120), -1)
        cv.line(frame, (x1, y1), (x2, y2), (0, 255, 220), 3)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv.putText(frame, f"{int(ema_dist)}px", (cx + 8, cy - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 220), 1)

        # ── Zoom Chrome ──────────────────────────────────────────────
        if (now - last_zoom_time) > ZOOM_COOLDOWN:
            if ema_dist > ZOOM_IN_THRESHOLD:
                zoom_chrome("in")
                last_zoom_time = now
                status_msg = "🔍 Zoom IN"
                status_timer = now
            elif ema_dist < ZOOM_OUT_THRESHOLD:
                zoom_chrome("out")
                last_zoom_time = now
                status_msg = "🔎 Zoom OUT"
                status_timer = now

        # ── Tab switch via middle finger ─────────────────────────────
        if is_middle_closed(lm) and (now - last_page_change_time) > PAGE_COOLDOWN:
            if lm[4].x < lm[8].x:
                switch_tab("prev")
                status_msg = "◀ Prev Tab"
            else:
                switch_tab("next")
                status_msg = "▶ Next Tab"
            last_page_change_time = now
            status_timer = now

        drawing.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        draw_ruler(frame, ema_dist)

    # ── HUD ──────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv.rectangle(overlay, (0, 0), (W, 70), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv.putText(frame, "GESTURE → CHROME", (20, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 180), 2)
    cv.putText(frame, "Q = quit", (W - 160, 40),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)

    # Status flash (shows for 1.5s)
    if status_msg and (now - status_timer) < 1.5:
        cv.putText(frame, status_msg, (W // 2 - 150, H // 2),
                   cv.FONT_HERSHEY_SIMPLEX, 1.6, (0, 200, 255), 3)

    cv.imshow("Gesture Control", frame)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()