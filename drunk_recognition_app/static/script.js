const uploadButton = document.getElementById('uploadButton');
const imageUpload = document.getElementById('imageUpload');
const predictionText = document.getElementById('predictionText');
const imageDisplay = document.getElementById('imageDisplay');
const result = document.getElementById('result');
const remoteCameraButton = document.getElementById('remoteCameraButton');

// Mobile menu functionality
document.addEventListener('DOMContentLoaded', function () {
    const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
    const mobileMenu = document.getElementById('mobile-menu');

    mobileMenuToggle.addEventListener('click', function () {
        // Toggle mobile menu
        if (mobileMenu.classList.contains('hidden')) {
            mobileMenu.classList.remove('hidden');
        } else {
            mobileMenu.classList.add('hidden');
        }
    });

    // Close mobile menu when clicking on links
    const mobileLinks = document.querySelectorAll('.mobile-link');
    mobileLinks.forEach(link => {
        link.addEventListener('click', function () {
            mobileMenu.classList.add('hidden');
        });
    });
});

// JavaScript to handle navigation link clicks
document.addEventListener('DOMContentLoaded', function () {
    const navLinks = document.querySelectorAll('nav ul li a');

    navLinks.forEach(link => {
        link.addEventListener('click', function (event) {
            event.preventDefault(); // Prevent default link behavior

            const targetId = this.getAttribute('href');

            if (targetId === 'index.html') {
                // Logic to navigate to the main page
                window.location.href = targetId;
            } else if (targetId === 'about.html') {
                // Logic to show about information
                window.location.href = targetId;
            }
        });
    });
});

// Loại bỏ đoạn setInterval kiểm tra file input (vì chúng ta sẽ xử lý khi bấm nút Predict)

// Khi bấm nút Predict, toàn bộ quy trình sẽ được gọi qua endpoint /predict_all
uploadButton.addEventListener('click', async () => {
    const file = imageUpload.files[0];
    if (!file) {
        alert('Please select an image.');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    try {
        // Gọi endpoint /predict_all thay vì /predict
        const response = await fetch('http://127.0.0.1:5000/predict_all', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            alert('Server error');
            return;
        }

        const data = await response.json();

        if (data.success) {
            // Cập nhật thông báo dự đoán
            predictionText.textContent = data.message;
            result.classList.remove('hidden');

            // Hiển thị ảnh đã được xử lý (được trả về dưới dạng base64)
            imageDisplay.src = `data:image/png;base64,${data.image_base64}`;
            imageDisplay.classList.remove('hidden');

            // Nếu cần lưu ảnh của trường hợp say, có thể gọi saveDrunkImage (nếu endpoint này vẫn cần thiết)
            if (data.is_drunk) {
                saveDrunkImage(file);
            }
        } else {
            alert('Prediction failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Connection error with the server.');
    }
});

remoteCameraButton.addEventListener('click', () => {
    alert('Remote camera feature has not been implemented yet.');
});

// Hàm saveDrunkImage (giữ nguyên nếu bạn muốn lưu ảnh khi dự đoán là drunk)
async function saveDrunkImage(file) {
    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('http://127.0.0.1:5000/save_image', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!data.success) {
            console.error('Failed to save image:', data.message);
        }
    } catch (error) {
        console.error('Error saving image:', error);
    }
}
