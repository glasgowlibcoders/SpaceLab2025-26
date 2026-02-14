from picamzero import Camera
from pathlib import Path
from time import sleep, time
from exif import Image
from datetime import datetime, timedelta
import cv2
import math
import numpy as np

gsd = 148

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "log.txt"
LOG_FILE = "log.txt"
PHOTOS_DIR = BASE_DIR / "photos"

start_time = time()


def log(msg):
    elapsed = time() - start_time
    elapsed_rounded = round(elapsed, 2)
    hours, remainder = divmod(elapsed_rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    timestamp = f"{int(minutes):02}:{seconds:05.2f}"

    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp} | {msg}\n")

log("START")

PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
cam = Camera()
captured_images = []
img_number = 0

# Capture images
while time() - start_time < 360:
    img_number += 1
    path = PHOTOS_DIR / f"image{img_number:03}.jpg"
    try:
        cam.take_photo(path)
        captured_images.append(path)
        log(f"Image taken {img_number}")
        log(f"Image capture success {img_number}")
        sleep(1)
    except Exception:
        log(f"Image capture failure {img_number}")
    sleep(2)

def get_time(path):
    try:
        with open(path, "rb") as f:
            img = Image(f)
            dt = img.get("datetime_original")
            if not dt:
                log(f"EXIF time missing {path.name}")
                return None
            dt_obj = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S")
            log(f"EXIF time read {path.name}")
            return dt_obj
    except Exception:
        log(f"EXIF read failed {path.name}")
        return None

def time_diff(a, b):
    t1 = get_time(a)
    t2 = get_time(b)
    if t1 is None or t2 is None:
        return None
    return (t2 - t1).total_seconds()

def to_cv(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        log(f"Image load failed {path.name}")
    else:
        log(f"Image load success {path.name}")
    return img

def features(a, b):
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, d1 = orb.detectAndCompute(a, None)
    kp2, d2 = orb.detectAndCompute(b, None)
    if d1 is None or d2 is None or kp1 is None or kp2 is None:
        log(f"Feature detection failed {a} or {b}")
    return kp1, kp2, d1, d2

def matches(d1, d2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(d1, d2, k=2)
    good = []
    for r in raw:
        if len(r) != 2:
            continue
        m, n = r
        if m.distance < 0.75 * n.distance:
            good.append(m)
    log(f"Match count {len(good)}")
    return good

def coords(kp1, kp2, m):
    a, b = [], []
    for x in m:
        a.append(kp1[x.queryIdx].pt)
        b.append(kp2[x.trainIdx].pt)
    return a, b

def top_distance(a, b):
    dx, dy, d = [], [], []

    for (x1, y1), (x2, y2) in zip(a, b):
        ddx = x2 - x1
        ddy = y2 - y1
        dist = math.hypot(ddx, ddy)
        if 2 <= dist <= 300:
            dx.append(ddx)
            dy.append(ddy)
            d.append(dist)

    if len(d) < 5:
        return 0.0

    mx = float(np.median(dx))
    my = float(np.median(dy))

    consistent = [
        dist
        for dist, ddx, ddy in zip(d, dx, dy)
        if abs(ddx - mx) < 5 and abs(ddy - my) < 5
    ]

    if len(consistent) < 5:
        return 0.0

    return float(np.median(consistent))

def speed(px, gsd, dt):
    if dt <= 0:
        log("Speed rejected - Invalid dt")
        return 0.0
    return (px * gsd / 1000.0) / dt

def filter_outliers(s):
    if len(s) < 4:
        return s
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    removed = [x for x in s if not (lo <= x <= hi)]
    for val in removed:
        log(f"Speed removed as outlier {val}")
    return [x for x in s if lo <= x <= hi]

pair_speeds = []

# Process pairs
for i in range(len(captured_images)):
    for j in range(i + 1, min(i + 3, len(captured_images))):

        if time() - start_time >= 585:
            log(f"Pair skipped - Timeout reached")
            break

        a = captured_images[i]
        b = captured_images[j]
        log(f"Pair started {a.name} - {b.name}")

        dt = time_diff(a, b)
        if dt is None or not (1.6 <= dt <= 5):
            log("Pair skipped - Invalid dt")
            continue

        img1 = to_cv(a)
        img2 = to_cv(b)
        if img1 is None or img2 is None:
            continue

        kp1, kp2, d1, d2 = features(img1, img2)
        if d1 is None or d2 is None or kp1 is None or kp2 is None:
            continue

        m = matches(d1, d2)
        if len(m) < 5:
            log("Pair rejected - Insufficient matches")
            continue

        c1, c2 = coords(kp1, kp2, m)
        dist = top_distance(c1, c2)
        if dist == 0:
            log("Speed rejected - Invalid distance")
            continue

        v = speed(dist, gsd, dt)
        pair_speeds.append(v)
        log(f"Speed calculated {a.name} - {b.name} - {v:.4f}")

    if time() - start_time >= 585:
        break

# Outlier handling
log("Outlier removal started")
final = filter_outliers(pair_speeds)
log(f"Speeds remaining {len(final)}")

# Final result
final_speed = float(np.median(final)) if final else 0.0
log(f"Final speed {final_speed:.5g}")

SAVED_PHOTOS=sorted((PHOTOS_DIR.iterdir()))
NUM_PHOTOS=len(SAVED_PHOTOS)
NUM_DELETE=NUM_PHOTOS - 42
if NUM_DELETE > 0
    for x in range(NUM_DELETE):
        SAVED_PHOTOS.pop()

try:
    with open(BASE_DIR / "result.txt", "w") as f:
        f.write(f"{final_speed:.5g}")
    log("Result file write success")
except Exception:
    log("Result file write failure")

# Shutdown
log("END")
log(f"Total runtime {time() - start_time:.2f}s")
