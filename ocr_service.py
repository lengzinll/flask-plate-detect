import base64
from flask import Flask, request, jsonify
import cv2
import numpy as np
import re
from paddleocr import PaddleOCR
from paddleocr.ppocr.utils.logging import get_logger
import logging

app = Flask(__name__)

logger = get_logger()
logger.setLevel(logging.ERROR)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, lang="en", save_crop_res=False)

@app.route('/ocr', methods=['POST'])
def ocr_plate():
    data = request.get_json()

    if 'images' not in data:
        return jsonify({"error": "No images provided"}), 400

    results = []
    for image_base64 in data['images']:
        # Decode the Base64 string back into bytes
        img_data = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            continue

        # Perform OCR
        ocr_result = ocr.ocr(img, det=False, rec=True, cls=False)
        text = ""
        for r in ocr_result:
            scores = r[0][1]
            if np.isnan(scores):
                scores = 0
            else:
                scores = int(scores * 100)
            if scores > 60:
                text = r[0][0]

        # Clean and format recognized text
        pattern = re.compile(r'[\W]')
        text = pattern.sub('', text)
        text = text.replace("???", "").replace("O", "0").replace("粤", "").replace("皖", "")
        if len(text) > 5:
            results.append({
                "license_plate": text
            })

    return jsonify(results)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
