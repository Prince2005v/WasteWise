WasteWise ♻️
AI-powered Waste Classification Web App

WasteWise ek Flask web application hai jo TensorFlow Lite aur MobileNetV2 transfer learning ka use karke waste images ko classify karta hai — Biodegradable, Recyclable, aur Landfill categories me.

🚀 Features:
📸 Upload waste images via web interface
🧠 AI-powered classification using MobileNetV2 + TrashNet dataset
📊 Real-time prediction results with disposal tips
🌐 API endpoint: /predict returns JSON responses
💾 Uploads folder for storing images
⚠️ Error handling for invalid uploads or model errors

🛠️ Tech Stack:
Frontend: HTML, CSS, JavaScript
Backend: Flask (Python)
AI Model: TensorFlow Lite (MobileNetV2) with transfer learning
Dataset: TrashNet Dataset

📂 Project Structure:
WasteWise/
│── app.py              # Flask main application
│── static/              # CSS, JS, images
│── templates/           # HTML templates
│── models/              # Trained TFLite model
│── uploads/             # Uploaded waste images
│── requirements.txt     # Python dependencies
│── README.md            # Project documentation

Visit http://127.0.0.1:5000/ in your browser.

📸 Future Improvements
📷 Live camera detection

📈 Waste disposal statistics dashboard

📦 Deploy to cloud (Heroku, AWS, etc.)
