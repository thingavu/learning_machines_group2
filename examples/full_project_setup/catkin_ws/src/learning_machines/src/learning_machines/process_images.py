import cv2
import numpy as np
import os
import time

from sympy.categories import Object

try:
    from data_files import RESULTS_DIR
except ImportError:
    RESULTS_DIR = "./results"
    print(
        f"Could not import RESULTS_DIR. Using fallback Directory {RESULTS_DIR}")

# Calibration constants HARDWARE
AREA_THRESHOLD = 50
GREEN_SATURATION_THRESHOLD = 60
RED_SATURATION_THRESHOLD = 100
DEBUG_FLAG = True
DEBUG_INTERVAL = 3  # Save 1 in every 10 images processed

# AREA_THRESHOLD = 10
# GREEN_SATURATION_THRESHOLD = 60
# RED_SATURATION_THRESHOLD = 60
# DEBUG_FLAG = True
# DEBUG_INTERVAL = 3  # Save 1 in every 10 images processed

# Directory for debug images
DEBUG_DIR = os.path.join(RESULTS_DIR, "DEBUG_IMAGES")
if DEBUG_FLAG:
    os.makedirs(DEBUG_DIR, exist_ok=True)

# Global counter for debug images
debug_image_counter = 0


def preprocess_real_image(image, target_size=(512, 512)):
    target_height, target_width = target_size
    height, width = image.shape[:2]

    if height > target_height:
        start_row = height - target_height
        image = image[start_row:height, :]
    return cv2.resize(image, (target_width, target_height))


def detect_nearest_object(image, OBJECT_COLOR, area_threshold=AREA_THRESHOLD, debug_save=DEBUG_FLAG):
    """
    Detect the nearest object of color OBJECT_COLOR (green or red) in the image.
    """
    global debug_image_counter

    start_time = time.time()

    # Preprocess if needed
    if image.shape[:2] != (512, 512):
        image = preprocess_real_image(image, target_size=(512, 512))

    processed_image = image.copy()
    annotated_image = processed_image.copy()  # Create a copy for annotations

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if OBJECT_COLOR == "green":
        lower = np.array([36, GREEN_SATURATION_THRESHOLD, 25])
        upper = np.array([70, 255, 255])
    elif OBJECT_COLOR == "red":
        # fill this in
        lower = np.array([0, RED_SATURATION_THRESHOLD, 25])
        upper = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    color_pixel_count = np.sum(mask) / 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    lower_center = (width / 2, height)

    nearest_object_area = 0
    nearest_object_point = (None, None)  # Will store the adjusted point (centroid shifted down)
    min_distance = float('inf')
    considered_contours = []

    # Process each contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_threshold:
            continue

        considered_contours.append(contour)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        # Compute centroid
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Calculate equivalent circle radius from area
        radius = int(np.sqrt(area / np.pi)) if area > 0 else 5
        # Shift centroid downward by half the radius

        if OBJECT_COLOR == "red":
            shifted_cy = cy + radius // 2
        else:
            shifted_cy = cy

        # Compute distance from the shifted point to the lower center of the image
        distance = np.sqrt((cx - lower_center[0]) ** 2 + (shifted_cy - lower_center[1]) ** 2)

        if distance < min_distance:
            min_distance = distance
            nearest_object_point = (cx, shifted_cy)
            nearest_object_area = area

    distance_to_lower_center = min_distance if min_distance != float('inf') else None

    # Annotate the image
    overlay_texts = [
        f"color Count: {color_pixel_count}",
        f"Nearest Area: {nearest_object_area}",
        f"Point: {nearest_object_point}",
        f"Dist: {distance_to_lower_center}"
    ]
    y0, dy = 30, 30
    color_map = {
        "green": (0, 255, 0),
        "red": (0, 0, 255)
    }
    for i, text in enumerate(overlay_texts):
        y = y0 + i * dy
        cv2.putText(annotated_image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color_map[OBJECT_COLOR], 2, cv2.LINE_AA)

    # Draw all considered targets in yellow using shifted points
    for contour in considered_contours:
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        radius = int(np.sqrt(area / np.pi)) if area > 0 else 5
        shifted_cy = cy + radius // 2

        cv2.circle(annotated_image, (cx, shifted_cy), radius, (0, 255, 255), 2)
        cv2.putText(annotated_image, f"{int(area)}", (cx + radius + 5, shifted_cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Draw the nearest object in red at its adjusted point
    if nearest_object_point[0] is not None:
        cx, shifted_cy = nearest_object_point
        radius = int(np.sqrt(nearest_object_area / np.pi)) if nearest_object_area > 0 else 5
        cv2.circle(annotated_image, (cx, shifted_cy), radius, (0, 0, 255), 2)
        cv2.putText(annotated_image, f"Nearest Area: {int(nearest_object_area)}",
                    (cx + radius + 5, shifted_cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save debug image if conditions are met
    if debug_save:
        if OBJECT_COLOR == "green":
            debug_image_counter += 1 # only update on green to prevent issues
        if DEBUG_INTERVAL in [0, 1] or (debug_image_counter) % DEBUG_INTERVAL == 0:
            debug_path = os.path.join(DEBUG_DIR, f"debug_{debug_image_counter}_{OBJECT_COLOR}.jpg")
            cv2.imwrite(debug_path, annotated_image)

    duration = round((time.time() - start_time), 3)
    # print(f"detected {len(considered_contours)} contours in {duration} seconds)")
    return {
        'color_pixel_count': color_pixel_count,    # The name of this one changed
        'mask': mask,
        'nearest_object_area': nearest_object_area,
        'nearest_object_centroid': nearest_object_point,
        'distance_to_lower_center': distance_to_lower_center,
        'considered_contours': considered_contours,
        'processed_image': processed_image,
        'annotated_image': annotated_image
    }


if __name__ == "__main__":
    # TEST_DIR = os.path.join(RESULTS_DIR, "TEST_IMAGES")
    TEST_DIR = os.path.join("TEST_IMAGES")
    BATCH_SIZE = 5
    SCALE_FACTOR = 0.5

    image_files = sorted(
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith((".jpg", ".jpeg"))
    )
    total_images = len(image_files)
    print(f"Found {total_images} images to process.")

    for batch_start in range(0, total_images, BATCH_SIZE):
        batch_files = image_files[batch_start:batch_start + BATCH_SIZE]
        originals = []
        masks_bgr = []

        for img_file in batch_files:
            img_path = os.path.join(TEST_DIR, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load {img_file}")
                continue

            features = detect_nearest_object(image, "green", area_threshold=AREA_THRESHOLD)
            # features = detect_nearest_object(image, "red", area_threshold=AREA_THRESHOLD)
            annotated_image = features['annotated_image']

            print(f"{img_file}: color Count = {features['color_pixel_count']}, "
                  f"Nearest Area = {features['nearest_object_area']}, "
                  f"Nearest Centroid = {features['nearest_object_centroid']}, "
                  f"Distance to Lower Center = {features['distance_to_lower_center']}")

            # Use the annotated image for display
            original_resized = cv2.resize(annotated_image, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            mask_color = cv2.cvtColor(features['mask'], cv2.COLOR_GRAY2BGR)
            mask_resized = cv2.resize(mask_color, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)

            originals.append(original_resized)
            masks_bgr.append(mask_resized)

        if originals and masks_bgr:
            top_row = np.hstack(originals)
            bottom_row = np.hstack(masks_bgr)
            combined_display = np.vstack((top_row, bottom_row))

            cv2.imshow('Top: Annotated Images | Bottom: color Masks', combined_display)
            print("Press any key for next batch or 'q' to quit.")
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    cv2.destroyAllWindows()