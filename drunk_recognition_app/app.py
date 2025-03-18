from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import datetime
import cv2
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from backend.Workflow import TextRemovalPipeline, TextMaskGenerator, Inpainter

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load model TensorFlow (tải một lần khi chạy server để giảm thời gian load)
model = load_model("model/Drunkthermal_resnet.h5")

UPLOAD_FOLDER = "uploads"
DRUNK_FOLDER = "the_drunk"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DRUNK_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    # Đọc ảnh hồng ngoại
    infrared_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if infrared_image is None:
        raise ValueError("Không đọc được ảnh từ: " + image_path)

    # Chuẩn hóa giá trị pixel
    normalized_img = cv2.normalize(infrared_image, None, 0, 255, cv2.NORM_MINMAX)
    normalized_img = np.uint8(normalized_img)

    # Ánh xạ màu nhiệt
    thermal_image = cv2.applyColorMap(normalized_img, cv2.COLORMAP_INFERNO)

    # Chuyển đổi màu từ BGR sang RGB
    img_rgb = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB)
    
    # Chuyển đổi sang float32
    img_rgb = img_rgb.astype(np.float32)

    # Resize ảnh
    img_rgb = tf.keras.preprocessing.image.smart_resize(img_rgb, size=(256,256), interpolation='bicubic')
    
    # Chuẩn hóa ảnh: đưa về khoảng [0, 1]
    img_rgb -= img_rgb.min()
    img_rgb /= (img_rgb.max() - img_rgb.min() + 1e-7)  # thêm 1e-7 để tránh chia 0

    return img_rgb, thermal_image


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about.html')
def about():
    return render_template("about.html")

@app.route('/favicon.ico')
def favicon():
    return send_from_directory("static", "favicon.ico", mimetype="image/vnd.microsoft.icon")

@app.route('/predict_all', methods=['POST'])
def predict_all():
    """
    1) Lưu ảnh gốc
    2) Đổi màu hồng ngoại -> thermal
    3) Ghi đè file thermal để pipeline remove text
    4) Remove text (inpainting)
    5) Dự đoán drunk / not drunk
    6) Trả về JSON chứa (is_drunk, message, ảnh đã xử lý base64)
    """
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"})

    # Bước 1: Lưu ảnh gốc
    file = request.files['image']
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Bước 2: Đổi màu hồng ngoại -> thermal (numpy)
    img_rgb, thermal_bgr = preprocess_image(filepath)

    # Bước 3: Ghi đè file (hoặc ghi file mới) với thermal
    #    - Ở pipeline remove text, nó sẽ đọc file từ ổ đĩa
    cv2.imwrite(filepath, thermal_bgr)  # Ghi đè file: bgr format

    # Bước 4: Remove text & Inpainting
    text_mask_gen = TextMaskGenerator(
        confidence_threshold=0.3,
        padding=6,
        padding_top=5,
        padding_bottom=3,
        secure_mode=True
    )
    inpainter = Inpainter(device="cuda", target_size=(512, 512), secure_mode=True)
    pipeline = TextRemovalPipeline(text_mask_gen, inpainter)

    # pipeline.process_image() sẽ tạo mask, inpaint và lưu ảnh kết quả
    final_path = pipeline.process_image(filepath)
    
    # Bước 5: Dự đoán drunk / not drunk
    #    - Dùng ảnh đã chuẩn hoá (img_rgb) cho model
    #      (vì ta đã preprocess ở Bước 2)
    prediction = model.predict(np.expand_dims(img_rgb, axis=0))
    is_drunk = bool(prediction[0][0] > 0.5)
    message = "This person is drunk." if is_drunk else "This person is not drunk."

    # Bước 6: Trả về JSON chứa thông tin + ảnh final base64
    # Đọc file ảnh đã inpaint xong -> base64
    with open(final_path, "rb") as f:
        img_bytes = f.read()
    encoded_image = base64.b64encode(img_bytes).decode("utf-8")

    return jsonify({
        "success": True,
        "is_drunk": is_drunk,
        "message": message,
        "image_base64": encoded_image
    })

if __name__ == '__main__':
    app.run(debug=True)
