import os
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io
import uuid
from model_utils import WasteClassifier, get_disposal_tips

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key_for_development")

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Initialize the waste classifier
classifier = WasteClassifier()

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_for_prediction(image_file):
    """Process uploaded image for model prediction."""
    try:
        # Open and process the image
        image = Image.open(image_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (224x224 for MobileNetV2)
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return None

@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and redirect to results."""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique filename
            if file.filename:
                filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            else:
                filename = str(uuid.uuid4()) + '.jpg'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process image for prediction
            file.seek(0)  # Reset file pointer
            processed_image = process_image_for_prediction(file)
            
            if processed_image is None:
                flash('Error processing image. Please try a different image.', 'error')
                return redirect(url_for('index'))
            
            # Get prediction
            prediction_result = classifier.predict(processed_image)
            
            if prediction_result is None:
                flash('Error classifying image. Please try again.', 'error')
                return redirect(url_for('index'))
            
            # Get disposal tips
            tips = get_disposal_tips(prediction_result['category'])
            
            return render_template('result.html', 
                                 prediction=prediction_result,
                                 tips=tips,
                                 filename=filename)
            
        except Exception as e:
            logging.error(f"Error during upload/prediction: {str(e)}")
            flash('An error occurred while processing your image. Please try again.', 'error')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload a PNG, JPG, JPEG, GIF, or WebP image.', 'error')
        return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict_api():
    """API endpoint for image classification."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG, GIF, WebP'}), 400
        
        # Process image for prediction
        processed_image = process_image_for_prediction(file)
        
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 400
        
        # Get prediction
        prediction_result = classifier.predict(processed_image)
        
        if prediction_result is None:
            return jsonify({'error': 'Error classifying image'}), 500
        
        # Get disposal tips
        tips = get_disposal_tips(prediction_result['category'])
        
        # Save file (optional for API)
        if file.filename:
            filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        else:
            filename = str(uuid.uuid4()) + '.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.seek(0)  # Reset file pointer
        file.save(filepath)
        
        response = {
            'success': True,
            'prediction': prediction_result,
            'disposal_tips': tips,
            'filename': filename
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error in predict API: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': classifier.model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
