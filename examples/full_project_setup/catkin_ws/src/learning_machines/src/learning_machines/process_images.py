import cv2
import numpy as np
import os

def detect_nearest_green_object(image, area_threshold=10000):
    """
    Detects green objects in the image and identifies the object whose centroid
    is closest to the lower center of the image.

    Returns a dictionary with:
    - green_pixel_count: Total count of green pixels.
    - mask: Binary mask of detected green regions.
    - nearest_object_area: Area of the green contour closest to the lower center.
    - nearest_object_centroid: (cx, cy) coordinates of that object's centroid.
    - distance_to_lower_center: Euclidean distance from the object's centroid to the lower center of the image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([70, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixel_count = np.sum(mask) / 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = image.shape[:2]
    lower_center = (width / 2, height)

    nearest_object_area = 0
    nearest_object_centroid = (None, None)
    min_distance = float('inf')

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        distance = np.sqrt((cx - lower_center[0]) ** 2 + (cy - lower_center[1]) ** 2)

        if distance < min_distance:
            min_distance = distance
            nearest_object_centroid = (cx, cy)
            nearest_object_area = cv2.contourArea(contour)

    distance_to_lower_center = min_distance if min_distance != float('inf') else None

    return {
        'green_pixel_count': green_pixel_count,
        'mask': mask,
        'nearest_object_area': nearest_object_area,
        'nearest_object_centroid': nearest_object_centroid,
        'distance_to_lower_center': distance_to_lower_center
    }


if __name__ == "__main__":
    RESULTS_DIR = "./results"

    def process_images_in_batches(batch_size=5, scale_factor=0.5):
        image_files = sorted(f for f in os.listdir(RESULTS_DIR) if f.startswith("step_") and f.endswith(".jpg"))
        total_images = len(image_files)
        print(f"Found {total_images} images to process.")

        for batch_start in range(0, total_images, batch_size):
            batch_files = image_files[batch_start:batch_start + batch_size]
            originals = []
            masks_bgr = []

            for img_file in batch_files:
                img_path = os.path.join(RESULTS_DIR, img_file)
                image = cv2.imread(img_path)

                if image is None:
                    print(f"Failed to load {img_file}")
                    continue

                features = detect_nearest_green_object(image)
                print(f"{img_file}: Green Count = {features['green_pixel_count']}, "
                      f"Nearest Area = {features['nearest_object_area']}, "
                      f"Nearest Centroid = {features['nearest_object_centroid']}, "
                      f"Distance to Lower Center = {features['distance_to_lower_center']}")

                # Draw the centroid on the original image if available
                if features['nearest_object_centroid'][0] is not None:
                    cv2.circle(image, features['nearest_object_centroid'], 10, (0, 0, 255), 2)

                # Resize images for batch display
                original_resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
                mask_color = cv2.cvtColor(features['mask'], cv2.COLOR_GRAY2BGR)
                mask_resized = cv2.resize(mask_color, None, fx=scale_factor, fy=scale_factor)

                originals.append(original_resized)
                masks_bgr.append(mask_resized)

            # Create display grid if we have images in the batch
            if originals and masks_bgr:
                top_row = np.hstack(originals) if len(originals) > 0 else None
                bottom_row = np.hstack(masks_bgr) if len(masks_bgr) > 0 else None

                if top_row is not None and bottom_row is not None:
                    combined_display = np.vstack((top_row, bottom_row))
                elif top_row is not None:
                    combined_display = top_row
                elif bottom_row is not None:
                    combined_display = bottom_row
                else:
                    continue

                cv2.imshow('Top: Originals with Centroid | Bottom: Green Masks', combined_display)
                print("Press any key for next batch or 'q' to quit.")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break

        cv2.destroyAllWindows()

    process_images_in_batches()
