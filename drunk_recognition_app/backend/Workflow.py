import os
import cv2
import numpy as np
import torch
import warnings
from PIL import Image

import mmcv
from mmocr.apis import MMOCRInferencer
from mmagic.apis import MMagicInferencer
from mmengine import Config

warnings.filterwarnings("ignore", category=UserWarning)

class TextMaskGenerator:
    def __init__(self, confidence_threshold=0.5, padding=5, padding_top=0, padding_bottom=0, secure_mode=True):
        """
        Khởi tạo TextMaskGenerator với các tham số tùy chỉnh và biến secure_mode để tăng tính bảo mật.
        
        Args:
            confidence_threshold (float): Ngưỡng tin cậy cho OCR.
            padding (int): Padding chung theo chiều ngang.
            padding_top (int): Padding bổ sung ở phía trên.
            padding_bottom (int): Padding bổ sung ở phía dưới.
            secure_mode (bool): Nếu True, sẽ kiểm tra file đầu vào và định dạng.
        """
        self.confidence_threshold = confidence_threshold
        self.padding = padding
        self.padding_top = padding_top
        self.padding_bottom = padding_bottom
        self.secure_mode = secure_mode
        
        # Chỉ cho phép các định dạng file ảnh nhất định
        self.allowed_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        
        # Load mô hình OCR với mô hình cải tiến nếu có thể
        try:
            self.ocr = MMOCRInferencer(det='DB_r50_char', rec='SATRN')
        except Exception as e:
            print(f"Warning: Không load được mô hình cải tiến, sử dụng fallback: {e}")
            self.ocr = MMOCRInferencer(det='DB_r18', rec='ABINet')
    
    def _validate_file(self, file_path):
        if self.secure_mode:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
            if not file_path.lower().endswith(self.allowed_extensions):
                raise ValueError(f"Định dạng file không được phép: {file_path}")
    
    def generate_mask(self, img_path, output_mask_path=None, confidence_threshold=None, padding=None, padding_top=None, padding_bottom=None):
        """
        Phát hiện vùng chữ trong ảnh, vẽ polygon có padding và tạo mask nhị phân.
        
        Args:
            img_path (str): Đường dẫn ảnh gốc.
            output_mask_path (str, optional): Đường dẫn lưu mask. Mặc định thêm hậu tố '_mask.png'.
            confidence_threshold, padding, padding_top, padding_bottom: Tham số có thể override các giá trị mặc định.
            
        Returns:
            tuple: (mask_path, mask_image)
        """
        confidence_threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        padding = padding if padding is not None else self.padding
        padding_top = padding_top if padding_top is not None else self.padding_top
        padding_bottom = padding_bottom if padding_bottom is not None else self.padding_bottom
        
        self._validate_file(img_path)
        
        if output_mask_path is None:
            base_path = os.path.splitext(img_path)[0]
            output_mask_path = f"{base_path}_mask.png"
        
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ {img_path}")
        h, w, _ = img.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            result = self.ocr(img_path)
            if (not result) or ('predictions' not in result) or (not result['predictions']):
                print("Warning: Không có kết quả OCR!")
                cv2.imwrite(output_mask_path, mask)
                return output_mask_path, mask
            
            # Trích xuất polygons và điểm số
            polygons = []
            scores = []
            if 'det_polygons' in result['predictions'][0]:
                polygons = result['predictions'][0]['det_polygons']
                if 'det_scores' in result['predictions'][0]:
                    scores = result['predictions'][0]['det_scores']
                    if len(scores) < len(polygons):
                        scores.extend([1.0] * (len(polygons) - len(scores)))
                else:
                    scores = [1.0] * len(polygons)
            else:
                for item in result['predictions'][0]:
                    if 'polygon' in item:
                        polygons.append(item['polygon'])
                        scores.append(item.get('score', 1.0))
            
            # Vẽ polygon có padding trên mask
            for idx, polygon in enumerate(polygons):
                if idx < len(scores) and scores[idx] < confidence_threshold:
                    continue
                points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                if padding > 0 or padding_top > 0 or padding_bottom > 0:
                    padded_points = points.copy()
                    center = np.mean(points, axis=0)
                    for i, point in enumerate(padded_points):
                        vector = point - center
                        length = max(1.0, np.linalg.norm(vector))
                        # Áp dụng padding theo chiều ngang
                        padded_points[i][0] += int((vector[0] / length) * padding)
                        # Áp dụng padding theo chiều dọc với hiệu chỉnh riêng cho trên và dưới
                        if vector[1] < 0:
                            padded_points[i][1] += int((vector[1] / length) * padding - padding_top)
                        else:
                            padded_points[i][1] += int((vector[1] / length) * padding + padding_bottom)
                    cv2.fillPoly(mask, [padded_points], 255)
                else:
                    cv2.fillPoly(mask, [points], 255)
            
            # Nối các vùng gần nhau bằng phép đóng hình thái học
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        except Exception as e:
            print(f"Lỗi trong quá trình xử lý OCR: {e}")
        
        cv2.imwrite(output_mask_path, mask)
        print(f"Mask đã được lưu tại: {output_mask_path}")
        return output_mask_path, mask
    
    def visualize_mask(self, img_path, mask, output_path=None):
        """
        Trực quan hóa mask bằng cách overlay lên ảnh gốc.
        
        Args:
            img_path (str): Đường dẫn ảnh gốc.
            mask (numpy.ndarray hoặc str): Mask nhị phân hoặc đường dẫn mask.
            output_path (str, optional): Đường dẫn lưu ảnh trực quan hóa.
            
        Returns:
            output_path (str): Đường dẫn lưu ảnh trực quan hóa.
        """
        self._validate_file(img_path)
        if output_path is None:
            base_path = os.path.splitext(img_path)[0]
            output_path = f"{base_path}_visualized.jpg"
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ {img_path}")
        
        if isinstance(mask, str):
            mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Không thể đọc mask từ {mask}")
        
        color_mask = np.zeros_like(img)
        color_mask[mask > 0] = [0, 0, 255]  # Màu đỏ
        
        alpha = 0.5
        overlay = cv2.addWeighted(img, 1, color_mask, alpha, 0)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, overlay)
        print(f"Ảnh trực quan hóa được lưu tại: {output_path}")
        return output_path
    
    def process_folder(self, input_folder, output_mask_folder=None, output_visualized_folder=None):
        """
        Xử lý tất cả các ảnh trong folder, tạo mask và trực quan hóa.
        """
        if not os.path.isdir(input_folder):
            raise NotADirectoryError(f"Folder không tồn tại: {input_folder}")
        
        if output_mask_folder is None:
            output_mask_folder = os.path.join(input_folder, "masks")
        if output_visualized_folder is None:
            output_visualized_folder = os.path.join(input_folder, "visualized")
        
        os.makedirs(output_mask_folder, exist_ok=True)
        os.makedirs(output_visualized_folder, exist_ok=True)
        
        for filename in os.listdir(input_folder):
            if not filename.lower().endswith(self.allowed_extensions):
                continue
            img_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            mask_out_path = os.path.join(output_mask_folder, f"{base_name}_mask.png")
            mask_path, mask = self.generate_mask(img_path, mask_out_path)
            vis_out_path = os.path.join(output_visualized_folder, f"{base_name}_visualized.jpg")
            self.visualize_mask(img_path, mask, vis_out_path)

class Inpainter:
    # Cache các mô hình để tránh tải lại nhiều lần
    deepfill_inpainter = None
    aot_gan_inpainter = None
    
    def __init__(self, device="cuda", target_size=None, secure_mode=True):
        """
        Khởi tạo Inpainter với cấu hình thiết bị và kích thước ảnh mục tiêu.
        
        Args:
            device (str): "cuda" hoặc "cpu".
            target_size (tuple): Kích thước ảnh sau khi resize (width, height), nếu None giữ nguyên.
            secure_mode (bool): Nếu True, sẽ kiểm tra file đầu vào.
        """
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.target_size = target_size
        self.secure_mode = secure_mode
        self.allowed_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    
    def _validate_files(self, *file_paths):
        if self.secure_mode:
            for path in file_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Không tìm thấy file: {path}")
                if not path.lower().endswith(self.allowed_extensions):
                    raise ValueError(f"Định dạng file không được phép: {path}")
    
    def perform_inpainting(self, img_path, mask_path, output_path):
        """
        Thực hiện inpainting (lấy xoá chữ) trên ảnh sử dụng mô hình MMagic với các phương pháp fallback.
        
        Args:
            img_path (str): Đường dẫn ảnh gốc.
            mask_path (str): Đường dẫn mask nhị phân.
            output_path (str): Đường dẫn lưu ảnh sau inpainting.
            
        Returns:
            bool: True nếu thành công, False nếu thất bại.
        """
        self._validate_files(img_path, mask_path)
        
        print(f"Sử dụng thiết bị: {self.device}")
        img_bgr = cv2.imread(img_path)
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or mask_gray is None:
            raise ValueError("Không thể đọc ảnh hoặc mask.")
        
        if self.target_size is not None:
            img_bgr = cv2.resize(img_bgr, self.target_size, interpolation=cv2.INTER_AREA)
            mask_gray = cv2.resize(mask_gray, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Lưu ảnh tạm thời cho inpainting
        temp_dir = os.path.dirname(output_path)
        os.makedirs(temp_dir, exist_ok=True)
        base_temp_name = os.path.splitext(os.path.basename(output_path))[0]
        temp_img_path = os.path.join(temp_dir, f"{base_temp_name}_temp_img.png")
        temp_mask_path = os.path.join(temp_dir, f"{base_temp_name}_temp_mask.png")
        cv2.imwrite(temp_img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(temp_mask_path, mask_binary)
        print("Đã lưu file tạm cho inpainting.")
        
        success = False
        
        # Phương pháp 1: DeepFillV1
        try:
            print("Thử inpainting với DeepFillV1...")
            if Inpainter.deepfill_inpainter is None:
                Inpainter.deepfill_inpainter = MMagicInferencer("deepfillv1", model_setting=1)
            result = Inpainter.deepfill_inpainter.infer(img=temp_img_path, mask=temp_mask_path, result_out_dir=output_path)
            if os.path.exists(output_path):
                print(f"Inpainting thành công! Ảnh lưu tại: {output_path}")
                success = True
        except Exception as e:
            print(f"DeepFillV1 thất bại: {e}")
        
        # Phương pháp 2: AOT-GAN
        if not success:
            try:
                print("Thử inpainting với AOT-GAN...")
                if Inpainter.aot_gan_inpainter is None:
                    Inpainter.aot_gan_inpainter = MMagicInferencer(
                        "aot-gan",
                        model_path="https://download.openmmlab.com/mmediting/inpainting/aot_gan/aot-gan_512x512_4x12_places_20220509-6dbc2129.pth"
                    )
                result = Inpainter.aot_gan_inpainter.infer(img=temp_img_path, mask=temp_mask_path)
                if "result" in result and len(result["result"]) > 0:
                    result_img = result["result"][0]
                    cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                    print(f"Inpainting thành công với AOT-GAN! Ảnh lưu tại: {output_path}")
                    success = True
            except Exception as e:
                print(f"AOT-GAN thất bại: {e}")
        
        # Phương pháp 3: Inpainting bằng OpenCV
        if not success:
            try:
                print("Thử inpainting với OpenCV...")
                inpaint_telea = cv2.inpaint(img_bgr, mask_binary, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                inpaint_ns = cv2.inpaint(inpaint_telea, mask_binary, inpaintRadius=10, flags=cv2.INPAINT_NS)
                alpha = 0.7
                blended = cv2.addWeighted(inpaint_telea, 1 - alpha, inpaint_ns, alpha, 0)
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
                sharpened = cv2.filter2D(blended, -1, kernel)
                cv2.imwrite(output_path, sharpened)
                print(f"Inpainting (OpenCV) thành công! Ảnh lưu tại: {output_path}")
                success = True
            except Exception as e:
                print(f"Inpainting với OpenCV thất bại: {e}")
        
        # Xóa file tạm
        try:
            os.remove(temp_img_path)
            os.remove(temp_mask_path)
        except Exception as e:
            print(f"Cảnh báo: Không xoá được file tạm: {e}")
        
        if not success:
            print("Tất cả các phương pháp inpainting thất bại. Lưu ảnh gốc.")
            cv2.imwrite(output_path, img_bgr)
        
        return success
    
    def batch_inpainting(self, img_folder, mask_folder, output_folder):
        """
        Xử lý inpainting cho tất cả ảnh trong folder.
        """
        if not os.path.isdir(img_folder):
            raise NotADirectoryError(f"Folder ảnh không tồn tại: {img_folder}")
        os.makedirs(output_folder, exist_ok=True)
        
        for filename in os.listdir(img_folder):
            if filename.lower().endswith(self.allowed_extensions):
                img_path = os.path.join(img_folder, filename)
                base_name = os.path.splitext(filename)[0]
                mask_filename = f"{base_name}_mask.png"
                mask_path = os.path.join(mask_folder, mask_filename)
                if not os.path.exists(mask_path):
                    print(f"Không tìm thấy mask cho {filename}. Bỏ qua.")
                    continue
                output_path = os.path.join(output_folder, filename)
                print(f"\nXử lý ảnh: {filename}")
                try:
                    result = self.perform_inpainting(img_path, mask_path, output_path)
                    if result:
                        print(f"Inpainting thành công cho {filename}!")
                    else:
                        print(f"Inpainting thất bại cho {filename}. Ảnh gốc được lưu lại.")
                except Exception as e:
                    print(f"Lỗi khi xử lý {filename}: {e}")

class TextRemovalPipeline:
    """
    Pipeline tích hợp thực hiện luồng: vẽ polygon → tạo mask → lấy xoá chữ.
    """
    def __init__(self, text_mask_generator: TextMaskGenerator, inpainter: Inpainter):
        self.text_mask_generator = text_mask_generator
        self.inpainter = inpainter
    
    def process_image(self, img_path, mask_output_path=None, inpaint_output_path=None):
        """
        Xử lý một ảnh đơn: tạo mask, trực quan hóa và inpainting.
        
        Args:
            img_path (str): Đường dẫn ảnh gốc.
            mask_output_path (str, optional): Đường dẫn lưu mask.
            inpaint_output_path (str, optional): Đường dẫn lưu ảnh sau inpainting.
            
        Returns:
            inpaint_output_path (str): Đường dẫn ảnh sau khi loại bỏ chữ.
        """
        print(f"Xử lý ảnh: {img_path}")
        # Bước 1: Tạo mask với polygon được vẽ
        mask_path, mask = self.text_mask_generator.generate_mask(img_path, output_mask_path=mask_output_path)
        # Tùy chọn: trực quan hóa mask
        self.text_mask_generator.visualize_mask(img_path, mask, output_path=None)
        
        # Bước 2: Inpainting để lấy xoá chữ
        if inpaint_output_path is None:
            base_path = os.path.splitext(img_path)[0]
            inpaint_output_path = f"{base_path}_inpainted.jpg"
        success = self.inpainter.perform_inpainting(img_path, mask_path, inpaint_output_path)
        
        if success:
            print(f"Xoá chữ thành công. Ảnh được lưu tại: {inpaint_output_path}")
        else:
            print("Xoá chữ thất bại.")
        return inpaint_output_path
    
    def process_folder(self, input_folder, mask_folder=None, visualized_folder=None, output_folder=None):
        """
        Xử lý toàn bộ ảnh trong folder thông qua pipeline.
        """
        if not os.path.isdir(input_folder):
            raise NotADirectoryError(f"Folder không tồn tại: {input_folder}")
        
        if mask_folder is None:
            mask_folder = os.path.join(input_folder, "masks")
        if visualized_folder is None:
            visualized_folder = os.path.join(input_folder, "visualized")
        if output_folder is None:
            output_folder = os.path.join(input_folder, "output")
        
        os.makedirs(mask_folder, exist_ok=True)
        os.makedirs(visualized_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(self.text_mask_generator.allowed_extensions):
                img_path = os.path.join(input_folder, filename)
                base_name = os.path.splitext(filename)[0]
                mask_output_path = os.path.join(mask_folder, f"{base_name}_mask.png")
                inpaint_output_path = os.path.join(output_folder, f"{base_name}_inpainted.jpg")
                
                # Tạo mask và trực quan hóa
                _, mask = self.text_mask_generator.generate_mask(img_path, output_mask_path=mask_output_path)
                self.text_mask_generator.visualize_mask(img_path, mask, output_path=os.path.join(visualized_folder, f"{base_name}_visualized.jpg"))
                
                # Thực hiện inpainting
                print(f"\nXử lý ảnh: {filename}")
                try:
                    result = self.inpainter.perform_inpainting(img_path, mask_output_path, inpaint_output_path)
                    if result:
                        print(f"Xoá chữ thành công cho {filename}!")
                    else:
                        print(f"Xoá chữ thất bại cho {filename}. Ảnh gốc được lưu lại.")
                except Exception as e:
                    print(f"Lỗi khi xử lý {filename}: {e}")

if __name__ == "__main__":
    # Chọn chế độ xử lý: "folder" cho toàn bộ folder, "single" cho một ảnh đơn.
    mode = "single"  # Thay đổi thành "single" nếu cần xử lý 1 ảnh
    
    # Khởi tạo các module với secure_mode=True để tăng tính bảo mật đầu vào
    text_mask_gen = TextMaskGenerator(confidence_threshold=0.3, padding=6, padding_top=5, padding_bottom=3, secure_mode=True)
    inpainter = Inpainter(device="cuda", target_size=(512, 512), secure_mode=True)
    pipeline = TextRemovalPipeline(text_mask_gen, inpainter)
    
    if mode == "folder":
        input_folder = r"D:\Fall 2024\AI Capstone Project\HT-03 White Hot\HT-03 White Hot"
        pipeline.process_folder(input_folder)
    elif mode == "single":
        img_path = r"D:\Fall 2024\AI Capstone Project\0004.jpg"
        pipeline.process_image(img_path)
