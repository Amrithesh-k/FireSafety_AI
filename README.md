# Title: SecureVision

## Overview:

SecureVision is a machine learning model designed to identify and alert residents and firefighters in the presence of fire or smoke. This README provides a comprehensive guide on the system, including its purpose, features, installation instructions, usage, and more.

## Features

### Fire and Smoke Detection:
The model uses advanced computer vision techniques to detect the presence of fire and smoke in images or video frames.
### Real-time Alerting:
The system provides real-time alerts to residents and firefighters when fire or smoke is detected.
### Integration with External Systems:
Easily integrate the system with existing alarm systems, notification services, or emergency response platforms.
### Customizable Thresholds:
Adjust detection thresholds based on specific environmental conditions or user preferences.
### Scalability:
The model is designed to scale efficiently, allowing deployment in various environments and settings.

## Usage
Web Interface:
Access the web interface at https://qwerty.streamlit.app/

![Image of the model working and detecting smoke](/smoke.jpeg)


## Installation
  
  ### Clone the Repository: 
  git clone https://github.com/Amritesh-K/SecureVision.git
  cd SecureVision

  ### Install Dependencies:
  pip install -r requirements.txt

  ### Install the pretrained model weights
  https://github.com/Amrithesh-k/FireSafety_AI/blob/main/data/trained_model_l.h5 download from this file and add it to the models directory 
  
  ### Run the Application:
  python app.py

  ## Model Architecture
  The Fire and Smoke Recognition model is built on ResNet-50 and VGG-16. The architecture is optimized for real-time detection and classification of fire and smoke in diverse environmental conditions.
