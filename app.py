from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import datetime
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # phải cài đúng tensorflow==2.17.1 với chạy được 

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load model TensorFlow (tải một lần khi chạy server để giảm thời gian load)
model = load_model("model\Drunk_spectrum_hot_best.h5")

UPLOAD_FOLDER = "uploads"
DRUNK_FOLDER = "the_drunk"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DRUNK_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    # Đọc ảnh hồng ngoại
    infrared_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Chuẩn hóa giá trị pixel
    normalized_img = cv2.normalize(infrared_image, None, 0, 255, cv2.NORM_MINMAX)
    normalized_img = np.uint8(normalized_img)

    # Ánh xạ màu nhiệt
    thermal_image = cv2.applyColorMap(normalized_img, cv2.COLORMAP_INFERNO)

    img = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB)
    
    # Chuyển đổi sang float32
    img = img.astype(np.float32)

    # Resize ảnh
    img = tf.keras.preprocessing.image.smart_resize(img, size=(256,256), interpolation='bicubic')
    
    # Chuẩn hóa
    img -= img.min()
    img /= (img.max() - img.min())

    return img

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/about.html")  # Thêm route này
def about():
    return render_template("about.html")

# Route để tự động phục vụ ảnh từ thư mục images
@app.route("/images/<path:filename>")
def serve_images(filename):
    return send_from_directory("images", filename)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory("static", "favicon.ico", mimetype="image/vnd.microsoft.icon")

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({"success": False, "message": "No image uploaded"})

#     file = request.files['image']
#     filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(filepath)

#     img = preprocess_image(filepath)
#     prediction = model.predict(np.expand_dims(img, axis=0))
 

    
#     is_drunk = bool(prediction[0][0] > 0.4)
#     message = "Người này say." if is_drunk else "Người này không say."
#     # in nguong cho tam hinh model predict


#     return jsonify({"success": True, "is_drunk": is_drunk, "message": message})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"})

    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = preprocess_image(filepath)
    prediction = model.predict(np.expand_dims(img, axis=0))

    # In ra ngưỡng (score) dự đoán
    threshold_score = float(prediction[0][0])
    print(f"Ngưỡng dự đoán: {threshold_score:.4f}")  # hoặc chỉ cần print(threshold_score)

    is_drunk = bool(threshold_score > 0.4)
    message = "Người này say." if is_drunk else "Người này không say."

    return jsonify({
        "success": True,
        "is_drunk": is_drunk,
        "message": message,
        "threshold": threshold_score  # Có thể trả về luôn nếu bạn muốn hiển thị trên client
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
    app.run(debug=True)
