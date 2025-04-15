const remoteCameraButton = document.getElementById('remoteCameraButton');
const captureButton = document.getElementById('captureButton');
const imageUpload = document.getElementById('imageUpload');
const uploadButton = document.getElementById('uploadButton');
const imageDisplay = document.getElementById('imageDisplay');
const predictionText = document.getElementById('predictionText');
const toggleCameraButton = document.getElementById('toggleCameraButton');
const result = document.getElementById('result');
const cameraWrapper = document.getElementById('cameraWrapper');
const video = document.getElementById('camera');

let videoStream = null;

// ===========================
// DOM Loaded
// ===========================
document.addEventListener('DOMContentLoaded', () => {
    // Mobile menu
    const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
    const mobileMenu = document.getElementById('mobile-menu');
    const mobileLinks = document.querySelectorAll('.mobile-link');

    mobileMenuToggle.addEventListener('click', () => {
        mobileMenu.classList.toggle('hidden');
    });

    mobileLinks.forEach(link => {
        link.addEventListener('click', () => {
            mobileMenu.classList.add('hidden');
        });
    });

    // Navigation links
    const navLinks = document.querySelectorAll('nav ul li a');
    navLinks.forEach(link => {
        link.addEventListener('click', event => {
            event.preventDefault();
            const targetId = link.getAttribute('href');
            window.location.href = targetId;
        });
    });
});

// ===========================
// Image Upload Handling
// ===========================
uploadButton.addEventListener('click', async () => {
    const file = imageUpload.files[0];
    if (!file) return alert('Please select an image.');

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.success) {
            imageDisplay.src = URL.createObjectURL(file);
            imageDisplay.classList.remove('hidden');
            result.classList.remove('hidden');
            predictionText.textContent = data.message;
            if (data.is_drunk) saveDrunkImage(imageDisplay.src);
        } else {
            alert('Prediction failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Connection error with the server.');
    }
});

// ===========================
// Camera Handling
// ===========================
remoteCameraButton.addEventListener('click', async () => {
    if (videoStream) {
        stopCamera();
    } else {
        await startCamera();
    }
});

toggleCameraButton.addEventListener('click', () => {
    stopCamera();
});

captureButton.addEventListener('click', () => {
    captureAndPredict();
});

async function startCamera() {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = videoStream;
        await video.play();

        cameraWrapper.style.display = 'block';
    } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Unable to access the camera.');
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
        video.srcObject = null;

        predictionText.textContent = "Camera stopped.";
        cameraWrapper.style.display = 'none';
        result.classList.add('hidden');
        imageDisplay.classList.add('hidden');
    }
}

// ===========================
// Capture from Camera
// ===========================
async function captureAndPredict() {
    if (!video.videoWidth || !video.videoHeight) {
        alert("Camera not ready yet.");
        return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageDataUrl = canvas.toDataURL('image/jpeg');
    const formData = new FormData();
    formData.append('image', dataURLtoBlob(imageDataUrl), 'capture.jpg');

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            imageDisplay.src = imageDataUrl;
            imageDisplay.classList.remove('hidden');
            result.classList.remove('hidden');
            predictionText.textContent = data.message;

            if (data.is_drunk) {
                saveDrunkImage(imageDataUrl);
            }
        } else {
            alert('Prediction failed. Please try again.');
        }
    } catch (error) {
        console.error('Error during prediction:', error);
        alert('Connection error with the server.');
    }
}

// ===========================
// Utility Functions
// ===========================
function dataURLtoBlob(dataURL) {
    const byteString = atob(dataURL.split(',')[1]);
    const mimeString = dataURL.split(',')[0].split(':')[1].split(';')[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: mimeString });
}

async function saveDrunkImage(imageDataUrl) {
    const formData = new FormData();
    formData.append('image', dataURLtoBlob(imageDataUrl));

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
