import cv2
import numpy as np
from ultralytics import YOLO


def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def draw_filled_rounded_rectangle(image, top_left, bottom_right, color, corner_radius):
    p1 = top_left
    p2 = (bottom_right[0], top_left[1])
    p3 = bottom_right
    p4 = (top_left[0], bottom_right[1])

    # Draw filled central rectangle
    central_rect_top_left = (p1[0] + corner_radius, p1[1])
    central_rect_bottom_right = (p3[0] - corner_radius, p3[1])
    cv2.rectangle(image, central_rect_top_left, central_rect_bottom_right, color, -1)

    # Draw filled side rectangles
    left_rect_top_left = (p1[0], p1[1] + corner_radius)
    left_rect_bottom_right = (p1[0] + corner_radius, p4[1] - corner_radius)
    cv2.rectangle(image, left_rect_top_left, left_rect_bottom_right, color, -1)

    right_rect_top_left = (p2[0] - corner_radius, p2[1] + corner_radius)
    right_rect_bottom_right = (p2[0], p4[1] - corner_radius)
    cv2.rectangle(image, right_rect_top_left, right_rect_bottom_right, color, -1)

    top_rect_top_left = (p1[0] + corner_radius, p1[1])
    top_rect_bottom_right = (p2[0] - corner_radius, p1[1] + corner_radius)
    cv2.rectangle(image, top_rect_top_left, top_rect_bottom_right, color, -1)

    bottom_rect_top_left = (p4[0] + corner_radius, p4[1] - corner_radius)
    bottom_rect_bottom_right = (p3[0] - corner_radius, p3[1])
    cv2.rectangle(image, bottom_rect_top_left, bottom_rect_bottom_right, color, -1)

    # Draw filled corner arcs
    cv2.ellipse(image, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180, 0, 90,
                color, -1)
    cv2.ellipse(image, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270, 0, 90,
                color, -1)
    cv2.ellipse(image, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0, 0, 90, color,
                -1)
    cv2.ellipse(image, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90, 0, 90, color,
                -1)


# Define the polygon coordinates
entrance_polygon = np.array([(943, 252), (703, 387), (1135, 696), (1264, 464)], np.int32)
entrance_polygon = entrance_polygon.reshape((-1, 1, 2))
counter_polygon = np.array([(206, 71), (198, 151), (289, 292), (456, 202), (270, 49)], np.int32)
counter_polygon = counter_polygon.reshape((-1, 1, 2))

model = YOLO("runs/detect/head_detection15/weights/best.pt")

# Open the video file
video_path = "HD CCTV Camera_cut.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# Custom color for this class
color = (255, 0, 255)
# Background color for the text
bg_color = (0, 0, 0)
class_color = tuple(map(int, color))
# Create a VideoWriter object to save the output video
videoWriter = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        result = model(frame, device='cuda:0', iou=0.5, conf=0.3)[0]

        # Visualize the results on the frame
        headcount_total = 0
        headcount_at_entrance = 0
        headcount_at_counter = 0
        sum_confidence = 0
        headcount_total += len(result.boxes)
        boxes = result.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), class_color, 2)

            # Calculate the center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            # Check if the center of the bounding box is inside the polygon
            if is_point_in_polygon((center_x, center_y), entrance_polygon):
                headcount_at_entrance += 1
            if is_point_in_polygon((center_x, center_y), counter_polygon):
                headcount_at_counter += 1

            # Confidence score
            confidence = box.conf[0]
            sum_confidence += confidence.item()
            confidence_text = f'{confidence:.2f}'
            # cv2.putText(frame, confidence_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Draw the polygon on the frame
        # cv2.polylines(frame, [entrance_polygon], isClosed=True, color=color, thickness=2)
        # cv2.polylines(frame, [counter_polygon], isClosed=True, color=color, thickness=2)

        # Put the headcount text on the frame
        customers_text = f'Customers total: {headcount_total}'
        entrance_text = f'Customers at entrance: {headcount_at_entrance}'
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1
        font_thickness = 1
        entrance_text_size = cv2.getTextSize(entrance_text, font, font_scale, font_thickness)[0]
        padding = 5
        text_x = width - entrance_text_size[0] - 3 * padding
        text_y = 50

        # Draw the background rectangle for polygon count text
        top_left = (text_x - padding, text_y - 30)
        bottom_right = (
            text_x + entrance_text_size[0] + padding, text_y + 3 * (entrance_text_size[1] + padding) + padding)
        corner_radius = 10
        draw_filled_rounded_rectangle(frame, top_left, bottom_right, bg_color, corner_radius)

        cv2.putText(frame, customers_text, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, entrance_text, (text_x, text_y + entrance_text_size[1] + padding),
                    font, font_scale, color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, f'Customers at counter: {headcount_at_counter}',
                    (text_x, text_y + 2 * (entrance_text_size[1] + padding)), font, font_scale, color,
                    font_thickness, cv2.LINE_AA)
        cv2.putText(frame, f'Mean confidence: {round(sum_confidence / len(result.boxes), 2)}',
                    (text_x, text_y + 3 * (entrance_text_size[1] + padding)),
                    font, font_scale, color, font_thickness, cv2.LINE_AA)

        videoWriter.write(frame)
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
videoWriter.release()
print(f"Video processing complete")
