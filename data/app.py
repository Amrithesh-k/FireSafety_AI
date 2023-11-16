import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import VideoTransformer, webrtc_streamer
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the trained model
trained_model_l = tf.keras.models.load_model("/mount/src/firesafety_ai/data/trained_model_l.h5")

# Load the label dictionary
label_dict = {0: "default", 1: "fire", 2: "smoke"}
IMG_SIZE = 224

# Define a VideoTransformer class to process the video frames
class VideoTransformer(VideoTransformer):
    def transform(self, frame):
        # Convert the frame to BGR format
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Your existing real-time detection code
        resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_for_pred = np.expand_dims(resized_frame, axis=0)
        frame_for_pred = preprocess_input(frame_for_pred)
        pred_vec = trained_model_l.predict(frame_for_pred)
        pred_class = []
        confidence = np.round(pred_vec.max(), 2)
        if confidence > 0.4:
            pc = np.argmax(pred_vec)
            pred_class.append((pc, confidence))
        else:
            pred_class.append((0, 0))
        if pred_class:
            txt = get_display_string(pred_class, label_dict)
            frame = draw_prediction(frame, txt)

        return frame

def draw_prediction(frame, class_string):
    x_start = frame.shape[1] - 600
    cv2.putText(frame, class_string, (x_start, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 2, cv2.LINE_AA)
    return frame

def get_display_string(pred_class, label_dict):
    txt = ""
    for c, confidence in pred_class:
        label = label_dict.get(int(c), "Unknown")
        txt += label
        if int(c):
            txt += '[' + str(confidence) + ']'
    return txt

def main():
    st.title("Real-time Detection App")
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
