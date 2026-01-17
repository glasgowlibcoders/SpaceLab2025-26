from picamzero import Camera
from pathlib import Path
from time import sleep
from exif import Image
from datetime import datetime, timedelta
import cv2
import math
import numpy as np
import sys

INTERVAL = 3
CAPTURE_DURATION = timedelta(seconds=18)
TOTAL_DURATION = timedelta(seconds=60)
MAX_IMAGES = 6
FEATURE_NUMBER = 300
COMPARE_LIMIT = 1
FILTER_OUTLIERS = True
MIN_TIME_DIFF = 3
MAX_TIME_DIFF = 6
TOP_PERCENTILE = 60
MAX_PIXEL_SHIFT = 600
ALTITUDE = 420_000
PIXEL_SIZE = 1.12e-6
FOCAL_LENGTH = 3.04e-3
GSD = ALTITUDE * PIXEL_SIZE / FOCAL_LENGTH

cam = Camera()
home_dir = Path.home() / "AstroPiImagesForTheMitchell"
home_dir.mkdir(exist_ok=True)

start_time = datetime.now()
capture_end = start_time + CAPTURE_DURATION
total_end = start_time + TOTAL_DURATION

captured_images = []

while (
    datetime.now() < capture_end
    and len(captured_images) < MAX_IMAGES
):
    if datetime.now() >= total_end:
        sys.exit("TIME LIMIT HIT DURING CAPTURE")

    filename = home_dir / f"image{len(captured_images)+1}.jpg"
    cam.take_photo(filename)
    captured_images.append(str(filename))

    sleep(INTERVAL)

def get_time(image_path):
    with open(image_path, "rb") as f:
        img = Image(f)
        return datetime.strptime(
            img.get("datetime_original"),
            "%Y:%m:%d %H:%M:%S"
        )

def get_time_difference(img1, img2):
    return (get_time(img2) - get_time(img1)).total_seconds()

def convert_to_cv(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def calculate_features(img1, img2):
    sift = cv2.SIFT_create(nfeatures=FEATURE_NUMBER)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    return kp1, kp2, des1, des2

def calculate_matches(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    return [m for m, n in matches if m.distance < 0.75 * n.distance]

def find_matching_coordinates(kp1, kp2, matches):
    coords1, coords2 = [], []
    for m in matches:
        coords1.append(kp1[m.queryIdx].pt)
        coords2.append(kp2[m.trainIdx].pt)
    return coords1, coords2

def calculate_top_percent_distance(coords1, coords2):
    distances = [
        math.hypot(x1 - x2, y1 - y2)
        for (x1, y1), (x2, y2) in zip(coords1, coords2)
        if math.hypot(x1 - x2, y1 - y2) <= MAX_PIXEL_SHIFT
    ]

    if not distances:
        return 0

    distances.sort(reverse=True)
    top_n = max(1, int(len(distances) * TOP_PERCENTILE / 100))
    return np.mean(distances[:top_n])

def calculate_speed(distance_px, time_s):
    if time_s <= 0:
        return 0
    return (distance_px * GSD / 1000) / time_s

def filter_outliers(values):
    if len(values) < 4:
        return values
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return [
        v for v in values
        if q1 - 1.5 * iqr <= v <= q3 + 1.5 * iqr
    ]

pair_speeds = []

for i in range(len(captured_images)):
    if datetime.now() >= total_end:
        sys.exit("TIME LIMIT HIT DURING PROCESSING")

    for j in range(i + 1, min(i + 1 + COMPARE_LIMIT, len(captured_images))):
        if datetime.now() >= total_end:
            sys.exit("TIME LIMIT HIT DURING PROCESSING")

        img1, img2 = captured_images[i], captured_images[j]
        dt = get_time_difference(img1, img2)

        if not (MIN_TIME_DIFF <= dt <= MAX_TIME_DIFF):
            continue

        img1_cv = convert_to_cv(img1)
        img2_cv = convert_to_cv(img2)

        kp1, kp2, des1, des2 = calculate_features(img1_cv, img2_cv)
        if des1 is None or des2 is None:
            continue

        matches = calculate_matches(des1, des2)
        if len(matches) < 5:
            continue

        coords1, coords2 = find_matching_coordinates(kp1, kp2, matches)
        dist = calculate_top_percent_distance(coords1, coords2)
        if dist == 0:
            continue

        pair_speeds.append(calculate_speed(dist, dt))

final_speeds = filter_outliers(pair_speeds) if FILTER_OUTLIERS else pair_speeds

median_speed = np.median(final_speeds) if final_speeds else 0
mean_speed = np.mean(final_speeds) if final_speeds else 0

result_file = Path.home() / "result.txt"
with open(result_file, "w") as f:
    f.write(f"Median: {median_speed:.5f} km/s\n")
    f.write(f"Mean: {mean_speed:.5f} km/s\n")

print(f"Estimated ISS speed (median): {median_speed:.5f} km/s")
print(f"Estimated ISS speed (mean):   {mean_speed:.5f} km/s")
print(f"Result saved to: {result_file}")
