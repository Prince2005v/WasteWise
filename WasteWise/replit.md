# Overview

Smart Waste Sorting System is a web-based AI application that helps users classify waste items into proper disposal categories (biodegradable, recyclable, or landfill) using image recognition. The system uses a MobileNetV2-based deep learning model to analyze uploaded images and provides disposal tips and recommendations. Built with Flask as the web framework and TensorFlow for machine learning capabilities, the application features a drag-and-drop interface for easy image uploads and real-time classification results.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Template Engine**: Jinja2 templates with Bootstrap for responsive UI
- **Styling**: Custom CSS with Bootstrap dark theme and Font Awesome icons
- **JavaScript**: Vanilla JavaScript for drag-and-drop functionality and form handling
- **Image Upload**: Supports multiple formats (PNG, JPG, JPEG, GIF, WebP) with 16MB size limit

## Backend Architecture
- **Web Framework**: Flask with session management and file upload handling
- **Model Architecture**: Transfer learning approach using pre-trained MobileNetV2 as base model
- **Classification Head**: Custom layers with GlobalAveragePooling2D, Dense layers, and Dropout for regularization
- **Image Processing**: PIL (Python Imaging Library) for image preprocessing and resizing to 224x224 pixels
- **File Management**: Secure filename handling with werkzeug utilities

## Machine Learning Pipeline
- **Model**: MobileNetV2-based convolutional neural network optimized for mobile deployment
- **Training Strategy**: Transfer learning with frozen base layers and custom classification head
- **Data Preprocessing**: Image normalization, resizing, and RGB conversion
- **Prediction**: Softmax output for three-class classification (biodegradable, recyclable, landfill)
- **Model Persistence**: HDF5 format for model storage and loading

## Data Storage
- **File Storage**: Local filesystem for uploaded images and model files
- **Directory Structure**: Organized folders for uploads and models with automatic creation
- **Session Management**: Flask sessions with configurable secret key

## Classification Logic
- **Waste Categories**: Three-tier classification system (biodegradable, recyclable, landfill)
- **Disposal Tips**: Integrated recommendation system providing category-specific guidance
- **Confidence Scoring**: Probability-based results with visual progress bars

# External Dependencies

## Core Frameworks
- **Flask**: Web application framework for routing, templating, and request handling
- **TensorFlow/Keras**: Deep learning framework for model creation, training, and inference
- **NumPy**: Numerical computing library for array operations and data manipulation

## Image Processing
- **PIL (Pillow)**: Python Imaging Library for image opening, conversion, and preprocessing
- **Werkzeug**: WSGI utilities for secure file handling and filename sanitization

## Frontend Libraries
- **Bootstrap**: CSS framework for responsive design and component styling
- **Font Awesome**: Icon library for user interface elements
- **Vanilla JavaScript**: Client-side functionality for drag-and-drop and form validation

## Development Tools
- **Matplotlib**: Plotting library for training visualization (used in training scripts)
- **Logging**: Python standard library for application logging and debugging

## Model Dependencies
- **ImageNet Weights**: Pre-trained weights from ImageNet dataset for transfer learning
- **MobileNetV2**: Lightweight CNN architecture optimized for mobile and edge devices