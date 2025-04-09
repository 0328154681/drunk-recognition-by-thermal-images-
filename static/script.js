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
document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('nav ul li a');

    navLinks.forEach(link => {
        link.addEventListener('click', function(event) {
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

// Check file input every 5 seconds
setInterval(() => {
    const file = imageUpload.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imageDisplay.src = e.target.result;
            simulateDrunkPrediction(file); 
        }
        reader.readAsDataURL(file);
    }
}, 5000);

uploadButton.addEventListener('click', async () => {
    const file = imageUpload.files[0];
    if (!file) {
        alert('Please select an image.');
        return;
    }

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
