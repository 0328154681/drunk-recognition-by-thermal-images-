from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import datetime
import cv2
import numpy as np
import tensorflow as tf
from flask_cors import CORS
import base64
import io
from PIL import Image
from tensorflow.keras.models import load_model  # Cần đúng version: tensorflow==2.17.1
import joblib
from ultralytics import YOLO


app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load filter model 
filter_model = joblib.load("model/spectrum_classifier.pkl")

# Load YOLO model
yolo_model = YOLO("model/yolo_model.pt")

# Load drunk_recognition model
model = load_model("model/Drunk_spectrum_hot_best.h5")


def is_spectrum_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img)

    def simple_entropy(ch):
        hist = cv2.calcHist([ch], [0], None, [32], [0, 256])
        hist = hist.ravel() / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    features = []
    for ch in [r, g, b]:
        features.append(np.mean(ch))
        features.append(np.std(ch))
        features.append(simple_entropy(ch))

    features = np.array(features).reshape(1, -1)

    pred = filter_model.predict(features)[0]
    print(f"🔍 Dự đoán filter_model:", pred, type(pred))  # debug

    return int(pred)  # đảm bảo là số nguyên 0 hoặc 1

def face_recognition(image_path):
    img = cv2.imread(image_path)
    results = yolo_model(img)

    pred_class = results[0].probs.top1  # 0 nếu no_face, 1 nếu has_face (tuỳ theo thư mục bạn đặt ban đầu)

    print(pred_class)
    
    return pred_class

def preprocess_image(image_path):
    # Đọc ảnh gốc (giữ nguyên màu như người dùng đưa vào)
    original_image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Chuyển sang float32 để đưa vào model
    img = img_rgb.astype(np.float32)

    # Resize ảnh theo đúng input của model
    img = tf.keras.preprocessing.image.smart_resize(img, size=(256, 256), interpolation='bicubic')

    # Chuẩn hóa về [0, 1]
    img -= img.min()
    img /= (img.max() - img.min())

    return img, img_rgb  # img: để predict, img_rgb: ảnh gốc giữ nguyên màu


UPLOAD_FOLDER = "uploads"
DRUNK_FOLDER = "the_drunk"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DRUNK_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route("/images/<path:filename>")
def serve_images(filename):
    return send_from_directory("images", filename)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory("static", "favicon.ico", mimetype="image/vnd.microsoft.icon")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"})

    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Bộ lọc trước khi predict
    spectrum_flag = is_spectrum_image(filepath)
    print(f"🔍 Dự đoán filter_model:", spectrum_flag, type(spectrum_flag))
    face_recog = face_recognition(filepath)
    print(f"👀 Face recognition:", face_recog, type(face_recog))
    if face_recog != 1:
        return jsonify({
            "success": True,
            "message": "Mô hình chỉ dự đoán trên ảnh nhiệt gương mặt. Vui lòng dùng ảnh nhiệt gương mặt.",
            "face_recognition": False
        }), 200  # vẫn trả HTTP 200 để không coi là lỗi phía client
    elif spectrum_flag != 1:
        return jsonify({
            "success": True,
            "message": "Ảnh không phải spectrum. Vui lòng dùng ảnh nhiệt.",
            "is_spectrum": False
        }), 200  # vẫn trả HTTP 200 để không coi là lỗi phía client
    
    img, original_image = preprocess_image(filepath)
    prediction = model.predict(np.expand_dims(img, axis=0))

    threshold_score = float(prediction[0][0])
    print(f"Ngưỡng dự đoán: {threshold_score:.4f}")  # hoặc chỉ cần print(threshold_score)

    is_drunk = bool(prediction[0][0] > 0.4)
    message = "Người này say." if is_drunk else "Người này không say."

    # Convert ảnh gốc thành base64
    image_for_display = Image.fromarray(original_image)  # original_image là RGB dạng uint8
    buffered = io.BytesIO()
    image_for_display.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({
        "success": True,
        "is_drunk": is_drunk,
        "message": message,
        "threshold": threshold_score,
        "image_base64": image_base64
    })


@app.route('/save_image', methods=['POST'])
def save_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"})

    file = request.files['image']
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    save_path = os.path.join(DRUNK_FOLDER, today)
    os.makedirs(save_path, exist_ok=True)

    file.save(os.path.join(save_path, file.filename))
    return jsonify({"success": True, "message": "Image saved successfully"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

