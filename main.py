from picamzero import Camera
from pathlib import Path
from time import sleep, time
from exif import Image
from datetime import datetime
import cv2
import math
import numpy as np

gsd = 146

BASE_DIR = Path(__file__).resolve().parent
cam = Camera()
start_time = time()

captured_images = []
img_number = 0

while time() - start_time < 360:
    img_number += 1
    path = BASE_DIR / f"image{img_number}.jpg"
    cam.take_photo(path)
    captured_images.append(path)
    sleep(2)

def get_time(path):
    try:
        with open(path, "rb") as f:
            img = Image(f)
            dt = img.get("datetime_original")
            if not dt:
                return None
            return datetime.strptime(dt, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None

def time_diff(a, b):
    t1 = get_time(a)
    t2 = get_time(b)
    if t1 is None or t2 is None:
        return None
    return (t2 - t1).total_seconds()

def to_cv(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

def features(a, b):
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, d1 = orb.detectAndCompute(a, None)
    kp2, d2 = orb.detectAndCompute(b, None)
    return kp1, kp2, d1, d2

def matches(d1, d2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(d1, d2, k=2)
    return [m for m, n in raw if m.distance < 0.75 * n.distance]

def coords(kp1, kp2, m):
    a, b = [], []
    for x in m:
        a.append(kp1[x.queryIdx].pt)
        b.append(kp2[x.trainIdx].pt)
    return a, b

def top_distance(a, b):
    d = [math.hypot(x1 - x2, y1 - y2) for (x1, y1), (x2, y2) in zip(a, b)]
    d = [x for x in d if x <= 1000]
    if not d:
        return 0.0
    d.sort(reverse=True)
    n = max(1, len(d) * 75 // 100)
    return float(np.mean(d[:n]))

def speed(px, gsd, dt):
    if dt <= 0:
        return 0.0
    return (px * gsd / 1000.0) / dt

def filter_outliers(s):
    if len(s) < 4:
        return s
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return [x for x in s if lo <= x <= hi]

pair_speeds = []

for i in range(len(captured_images)):
    for j in range(i + 1, min(i + 3, len(captured_images))):

        if time() - start_time >= 585:
            break

        a = captured_images[i]
        b = captured_images[j]

        dt = time_diff(a, b)
        if dt is None or not (1.6 <= dt <= 5):
            continue

        img1 = to_cv(a)
        img2 = to_cv(b)

        kp1, kp2, d1, d2 = features(img1, img2)
        if d1 is None or d2 is None:
            continue

        m = matches(d1, d2)
        if len(m) < 5:
            continue

        c1, c2 = coords(kp1, kp2, m)
        dist = top_distance(c1, c2)
        if dist == 0:
            continue

        pair_speeds.append(speed(dist, gsd, dt))

    if time() - start_time >= 585:
        break

final = filter_outliers(pair_speeds)
mean_speed = float(np.mean(final)) if final else 0.0

with open(BASE_DIR / "result.txt", "w") as f:
    f.write(f"{mean_speed:.5g}")