import cv2
import numpy as np

def main():
    template_path = 'sample.png'
    test_image_path = 'test.png'
    output_image_path = 'annotated_image.png'
    output_text_path = 'location_table.txt'

    similarity_thresh = 0.3
    min_area_thresh = 500

    print("Loading template and test images...")
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test_img_color = cv2.imread(test_image_path)
    if template is None or test_img_color is None:
        print("Failed to read images. Please check the file paths.")
        return

    test_img_gray = cv2.cvtColor(test_img_color, cv2.COLOR_BGR2GRAY)
    # Binarization
    _, template_thresh = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, test_thresh = cv2.threshold(test_img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract template contour
    contours_template, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_template:
        print("No template contour found.")
        return
    template_contour = max(contours_template, key=cv2.contourArea)

    # Extract test image contours
    contours_test, _ = cv2.findContours(test_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_objects = []
    print("Starting shape matching...")
    for cnt in contours_test:
        area = cv2.contourArea(cnt)
        if area < min_area_thresh:
            continue

        similarity = cv2.matchShapes(template_contour, cnt, cv2.CONTOURS_MATCH_I1, 0.0)
        if similarity < similarity_thresh:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            angle = rect[-1]
            (w, h) = rect[1]
            # Adjust angle based on width-height relationship
            if w < h:
                adjusted_angle = angle
            else:
                adjusted_angle = angle + 90

            found_objects.append({
                'box': box,
                'center': (cx, cy),
                'angle': adjusted_angle,
                'initial_angle': adjusted_angle,  # save initial angle
                'size': (w, h),
                'area': w * h  # calculate area from bounding box
            })

    print(f"Found {len(found_objects)} similar targets before size filtering.")

    # 计算平均面积并过滤大小明显异常的工件
    if found_objects:
        areas = [obj['area'] for obj in found_objects]
        average_area = np.mean(areas)
        # 设定面积偏差阈值，例如偏离平均值80%以上则认为异常
        deviation_threshold = 0.8
        filtered_objects = []
        for obj in found_objects:
            if abs(obj['area'] - average_area) / average_area <= deviation_threshold:
                filtered_objects.append(obj)
        print(f"Filtered out {len(found_objects) - len(filtered_objects)} targets due to size deviation.")
        found_objects = filtered_objects

    print(f"{len(found_objects)} targets remain after size filtering.")

    # Initial annotation on the image
    for idx, obj in enumerate(found_objects, start=1):
        cv2.drawContours(test_img_color, [obj['box']], 0, (0, 0, 255), 2)
        cv2.circle(test_img_color, obj['center'], 5, (0, 255, 0), -1)
        cv2.putText(test_img_color, f"{obj['angle']:.1f}°", 
                    (obj['center'][0] + 10, obj['center'][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        center = obj['center']
        angle = obj['angle']
        length = 50
        rad = np.radians(angle)
        end_point = (int(center[0] + length * np.cos(rad)),
                     int(center[1] + length * np.sin(rad)))
        cv2.arrowedLine(test_img_color, center, end_point, (0, 255, 0), 2, tipLength=0.2)

        offset_x = -50
        offset_y = 0
        id_position = (center[0] + offset_x, center[1] + offset_y)
        cv2.putText(test_img_color, f"ID:{idx}", id_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Use difference method to correct angles
    print("Starting angle correction...")
    template_h, template_w = template.shape
    t_center = (template_w // 2, template_h // 2)
    search_range = 5    # degrees
    step = 1            # degree step

    for obj in found_objects:
        cx, cy = obj['center']
        initial_angle = obj['angle']
        w, h = obj['size']
        min_diff = float('inf')
        best_angle = initial_angle

        # Search in vicinity of initial angle
        for candidate_angle in np.arange(initial_angle - search_range, initial_angle + search_range + step, step):
            M = cv2.getRotationMatrix2D(t_center, candidate_angle, 1.0)
            rotated_template = cv2.warpAffine(template, M, (template_w, template_h))
            start_x = t_center[0] - int(w/2)
            start_y = t_center[1] - int(h/2)
            # Ensure cropping area is within template bounds
            if start_x < 0 or start_y < 0 or start_x+int(w) > template_w or start_y+int(h) > template_h:
                continue
            cropped_template = rotated_template[start_y:start_y+int(h), start_x:start_x+int(w)]

            region_start_x = int(cx - w/2)
            region_start_y = int(cy - h/2)
            # Ensure target region is within image bounds
            if region_start_x < 0 or region_start_y < 0 or region_start_x+int(w) > test_img_gray.shape[1] or region_start_y+int(h) > test_img_gray.shape[0]:
                continue
            region = test_img_gray[region_start_y:region_start_y+int(h), region_start_x:region_start_x+int(w)]

            if cropped_template.shape != region.shape:
                continue
            diff = cv2.absdiff(region, cropped_template)
            diff_val = np.sum(diff)
            if diff_val < min_diff:
                min_diff = diff_val
                best_angle = candidate_angle

        obj['angle'] = best_angle

    # Redraw corrected angles on image
    for obj in found_objects:
        center = obj['center']
        angle = obj['angle']
        # Update angle display with corrected value
        cv2.putText(test_img_color, f"{angle:.1f}°", 
                    (center[0] + 10, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        length = 50
        rad = np.radians(angle)
        end_point = (int(center[0] + length * np.cos(rad)),
                     int(center[1] + length * np.sin(rad)))
        cv2.arrowedLine(test_img_color, center, end_point, (255, 0, 255), 2, tipLength=0.2)

    cv2.imwrite(output_image_path, test_img_color)
    print(f"Annotated image saved as '{output_image_path}'.")

    with open(output_text_path, 'w', encoding='utf-8') as f:
        for idx, obj in enumerate(found_objects, start=1):
            f.write(f"Target {idx}: Center {obj['center']}, Recognition angle {obj['initial_angle']:.2f}°, Corrected angle {obj['angle']:.2f}°, Border coordinates {obj['box'].tolist()}\n")
    print(f"Detection results written to '{output_text_path}'.")

if __name__ == "__main__":
    main()
