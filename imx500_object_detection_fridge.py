import argparse
import sys
import time
import json
import requests
from functools import lru_cache

import cv2
import numpy as np
import RPi.GPIO as GPIO

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

# Global variables for detections and product lists
last_detections = []
last_results = None
latest_in_products = []
latest_out_products = []
record_detections = False  # Set to True when motion is detected

# ----------------------
# PIR Sensor Setup
# ----------------------
PIR_PIN = 17  # Your PIR sensor GPIO pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

# ----------------------
# Object Detection Setup
# ----------------------
class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Store detection info and convert the bounding box to image coordinates."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def parse_detections(metadata: dict):
    """Convert network outputs into detection objects."""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0], conf=threshold, iou_thres=iou,
            max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h
        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_detections

@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def draw_detections(request, stream="main"):
    """
    Draw detection bounding boxes and update persistent product lists.
    Items are categorized as "in" or "out" based on their vertical position 
    relative to a horizontal dividing line. When motion is detected, a product 
    is added to the respective list and removed from the other list if it exists.
    """
    global latest_in_products, latest_out_products, last_results, record_detections
    detections = last_results
    if detections is None:
        return
    labels = get_labels()
    with MappedArray(request, stream) as m:
        frame_h, frame_w = m.array.shape[:2]
        horizontal_center = frame_h // 2

        # Draw the horizontal dividing line.
        cv2.line(m.array, (0, horizontal_center), (frame_w, horizontal_center), (255, 0, 0), 2)

        # Process each detection.
        for detection in detections:
            x, y, w, h = detection.box
            center_y = y + h // 2
            # Determine if the detection is "in" (above the center) or "out" (below the center).
            # Updated code: "out" if above the center, "in" if below
            position = "out" if center_y < horizontal_center else "in"

            label_text = f"{labels[int(detection.category)]} ({detection.conf:.2f}, {position})"

            # Only update persistent lists if detection recording is enabled.
            if record_detections:
                product = labels[int(detection.category)]
                if position == "in":
                    if product in latest_out_products:
                        latest_out_products.remove(product)
                    if product not in latest_in_products:
                        latest_in_products.append(product)
                elif position == "out":
                    if product in latest_in_products:
                        latest_in_products.remove(product)
                    if product not in latest_out_products:
                        latest_out_products.append(product)

            # Draw the bounding box and label.
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15
            overlay = m.array.copy()
            cv2.rectangle(overlay, (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255), cv2.FILLED)
            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)
            cv2.putText(m.array, label_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        # Optionally, draw a side panel showing the persistent product lists.
        if record_detections:
            panel_width = 150
            panel_start = frame_w - panel_width
            cv2.rectangle(m.array, (panel_start, 0), (frame_w, frame_h), (50, 50, 50), cv2.FILLED)
            y_offset = 20
            cv2.putText(m.array, "In Products:", (panel_start + 5, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            for prod in latest_in_products:
                cv2.putText(m.array, prod, (panel_start + 5, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            y_offset += 20
            cv2.putText(m.array, "Out Products:", (panel_start + 5, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            for prod in latest_out_products:
                cv2.putText(m.array, prod, (panel_start + 5, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), color, 2)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction,
                        help="Normalize bbox")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="Set bbox order: yx -> (y0, x0, y1, x1) or xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="IOU threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction,
                        help="Remove '-' labels")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Type of post processing")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="Preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Initialize the camera and network intrinsics.
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics with command-line parameters.
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    picam2.pre_callback = draw_detections

    # Variables to manage motion detection timing
    last_motion_time = time.time()

    # URL and headers for sending data via ngrok
    url = 'https://current_host_address.ngrok.app/message'
    headers = {"Content-Type": "application/json"}

    try:
        while True:
            current_time = time.time()

            # Check PIR sensor: 1 means motion detected, 0 means no motion.
            sensor_value = GPIO.input(PIR_PIN)
            if sensor_value == 1:
                print("Motion detected.")
                record_detections = True
                last_motion_time = current_time  # Update last motion time when motion is detected
            else:
                print("No motion.")

            # Update detections using the camera's metadata.
            last_results = parse_detections(picam2.capture_metadata())

            # If recording is active and no motion has been detected for 10 seconds, send the payload.
            if record_detections and (current_time - last_motion_time > 1):
                payload = {
                    "in": latest_in_products,   # Products detected as available
                    "out": latest_out_products  # Products detected as unavailable
                }
                try:
                    response = requests.post(url, data=json.dumps(payload), headers=headers)
                    print("Status code:", response.status_code)
                    print("Response:", response.text)
                except Exception as e:
                    print("Error sending data:", e)

                # Reset the recording state and product lists after sending.
                record_detections = False
                latest_in_products = []
                latest_out_products = []

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Measurement stopped by user.")

    finally:
        # Clean up GPIO settings for the PIR sensor.
        GPIO.cleanup()
