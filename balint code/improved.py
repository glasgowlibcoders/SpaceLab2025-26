from exif import Image as ExifImage
from datetime import datetime
import cv2
import math
import numpy as np
import glob
import os

def get_time(image_path):
    try:
        with open(image_path, 'rb') as f:
            img = ExifImage(f)
            time_str = img.get("datetime_original")
            if time_str is None:
                raise ValueError(f"No EXIF datetime for {image_path}")
            return datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    except Exception as e:
        print(f"Error reading time for {image_path}: {e}")
        return None

def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    if time_1 is None or time_2 is None:
        raise ValueError("Could not calculate time difference due to missing EXIF data")
    return (time_2 - time_1).total_seconds()

def load_and_preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img = cv2.equalizeHist(img)
    return img

def detect_features(image, feature_limit=2000):
    sift = cv2.SIFT_create(nfeatures=feature_limit)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(des1, des2):
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return sorted(good_matches, key=lambda x: x.distance)

def filter_outliers(coords1, coords2, threshold=50):
    if not coords1 or not coords2:
        return [], []
    distances = [math.hypot(x1-x2, y1-y2) for (x1, y1), (x2, y2) in zip(coords1, coords2)]
    median_distance = np.median(distances)
    filtered_coords1, filtered_coords2 = [], []
    for (c1, c2, d) in zip(coords1, coords2, distances):
        if abs(d - median_distance) < threshold:
            filtered_coords1.append(c1)
            filtered_coords2.append(c2)
    return filtered_coords1, filtered_coords2

def get_coordinates(kp1, kp2, matches):
    coords1, coords2 = [], []
    for m in matches:
        coords1.append(kp1[m.queryIdx].pt)
        coords2.append(kp2[m.trainIdx].pt)
    return coords1, coords2

def mean_distance(coords1, coords2):
    if not coords1 or not coords2:
        return None
    distances = [math.hypot(x1-x2, y1-y2) for (x1, y1), (x2, y2) in zip(coords1, coords2)]
    return np.mean(distances)

def calculate_speed_kmps(feature_distance, GSD, time_difference):
    if feature_distance is None or time_difference <= 0:
        return None
    distance_km = feature_distance * GSD / 1000  # meters to km
    speed_kmps = distance_km / time_difference
    return speed_kmps

def display_matches(image1, kp1, image2, kp2, matches, max_matches=100):
    img_matches = cv2.drawMatches(
        image1, kp1, image2, kp2, matches[:max_matches], None, flags=2
    )
    img_matches = cv2.resize(img_matches, (1600, 600), interpolation=cv2.INTER_AREA)
    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

GSD = 161  # meters per pixel 
IMAGE_FOLDER = r'C:\Users\User\Code_Things\AstroPy\results'
FEATURE_LIMIT = 2000
NUM_IMAGES = 10 

image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")))
image_paths = image_paths[:NUM_IMAGES]

speeds = []

for i in range(len(image_paths)-1):
    img1_path = image_paths[i]
    img2_path = image_paths[i+1]

    img1 = load_and_preprocess(img1_path)
    img2 = load_and_preprocess(img2_path)

    try:
        time_diff = get_time_difference(img1_path, img2_path)
        if time_diff <= 0:
            continue
    except:
        continue

    kp1, des1 = detect_features(img1, FEATURE_LIMIT)
    kp2, des2 = detect_features(img2, FEATURE_LIMIT)

    matches = match_features(des1, des2)

    coords1, coords2 = get_coordinates(kp1, kp2, matches)
    coords1, coords2 = filter_outliers(coords1, coords2)

    if not coords1 or not coords2:
        continue

    avg_distance = mean_distance(coords1, coords2)

    speed = calculate_speed_kmps(avg_distance, GSD, time_diff)
    if speed is not None:
        speeds.append(speed)

if speeds:
    overall_speed = np.mean(speeds)
    print(f"\nEstimated ISS speed over {len(speeds)} image pairs: {overall_speed:.6f} km/s")
else:
    print("No valid image pairs found to calculate speed.")