import cv2
import numpy as np
import base64
from datetime import datetime
import os
import re
import requests
import sys

api_endpoint = "http://localhost:3000/api/add-plate"

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def is_rtsp_url(string):
    rtsp_pattern = r"^(rtsp:\/\/)([^\/]*)(:(\d+))?(\/.*)?$"
    return re.match(rtsp_pattern, string) is not None

def find_car_from_plate(plate_image, frame_image):
    # Convert images to grayscale
    plate_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)

    # Template matching
    result = cv2.matchTemplate(frame_gray, plate_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8  # Adjust the threshold as needed
    loc = np.where(result >= threshold)

    # Process each detected plate
    for pt in zip(*loc[::-1]):
        # Get the coordinates of the detected plate
        x, y = pt
        h, w = plate_gray.shape
        
        # Define a larger area around the plate to extract the entire car
        horizontal_margin = 50  # Horizontal margin for left and right
        vertical_margin_top = 120  # Increased margin above the plate
        vertical_margin_bottom = 40  # Smaller margin below the plate
        
        car_x_start = max(x - horizontal_margin, 0)
        car_y_start = max(y - vertical_margin_top, 0)
        car_x_end = min(x + w + horizontal_margin, frame_image.shape[1])
        car_y_end = min(y + h + vertical_margin_bottom, frame_image.shape[0])
        
        # Draw a rectangle around the detected car on the original color image
        cv2.rectangle(frame_image, (car_x_start, car_y_start), (car_x_end, car_y_end), (0, 255, 0), 2)

    return frame_image

def image_to_base_64(img):
    _, buffer = cv2.imencode('.jpg', img)
    obj_image_base64 = base64.b64encode(buffer).decode('utf-8')
    return obj_image_base64

def save(location: str, license_plates: list[dict], start_time: datetime, end_time: datetime):
    data = []
    for plate in license_plates:
        new_obj = plate.copy()
        new_obj.update({
            'location' : location, 
            'start_time': start_time.isoformat(), 
            'end_time': end_time.isoformat(),
            "license_plate": str(plate["license_plate"]).upper(),
        })
        data.append(new_obj)

    # for i in data:
    #     print(i["license_plate"])

    if data:
        response = requests.post(api_endpoint, json=data)
        response.raise_for_status()
        print("save => ", response.text)