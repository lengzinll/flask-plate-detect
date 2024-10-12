import base64
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import requests
from threading import Thread
from helper import find_car_from_plate, image_to_base_64, save
from ultralytics import YOLO
from video import VideoStream
from datetime import datetime

app = Flask(__name__)

# Constants
OCR_URL = "http://localhost:5001/ocr"
SAVE_PERIOD = 15

license_plates = list()
all_license_plates = list()

model = YOLO("plate.pt")

def send_to_ocr(cropped_images, frame):
    # Step 1: Encode all images to send in one batch request
    encoded_images = []
    for img in cropped_images:
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        encoded_images.append(img_base64)

    if len(encoded_images) == 0:
        return
    
    # Step 2: Prepare a payload to send as JSON or multi-part form
    payload = { "images": encoded_images }

    try:
        response = requests.post(OCR_URL, json=payload)
        # Step 4: Process the OCR results
        if response.status_code == 200:
            ocr_results = response.json()  # Assuming the OCR API returns a list of results

            if not ocr_results:
                return

            for i, result in enumerate(ocr_results):
                plate_text = result.get('license_plate', '')
                frame_image = find_car_from_plate(cropped_images[i], frame)  # Match the original cropped image
                if not any(item['license_plate'] == plate_text for item in license_plates):
                    obj = {
                        "license_plate": plate_text,
                        "image": image_to_base_64(cropped_images[i]),
                        "frame": image_to_base_64(frame_image)
                    }
                    license_plates.append(obj)
                    all_license_plates.append(obj)

    except Exception as e:
        print(f"Error in OCR API: {e}")

def video_stream(camera_url: str):
    startTime = datetime.now()

    cap = VideoStream(camera_url).start()
    frame_count = 0

    while True:
        frame = cap.read()
        if frame is None:
            break
        currentTime = datetime.now()
        # frame = cv2.resize(frame, (1280, 720))

        frame_count += 1
        if frame_count % 5 == 0:
            results = model.predict(frame, conf=0.45, verbose=False)
            obj_images = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    obj_image = frame[y1:y2, x1:x2]
                    obj_images.append(obj_image)

            ocr_thread = Thread(target=send_to_ocr, args=(obj_images, frame))
            ocr_thread.start()

            if (currentTime - startTime).seconds >= SAVE_PERIOD:
                endTime = currentTime
                save_thread = Thread(target=save, args=("CAM1", license_plates, startTime, endTime))
                save_thread.start()
                startTime = currentTime
                license_plates.clear()

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame to stream it
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    camera_url = request.args.get('camera_url')
    return Response(video_stream(camera_url), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_detected_plates', methods=['GET'])
def get_detected_plates():
    global all_license_plates
    plates = all_license_plates[-50:]
    return jsonify({"license_plates": plates})  # Assuming license_plates is a global list



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,  debug=True)
