
# app.py
from camera_input_live import camera_input_live
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import math
import json
from matplotlib import pyplot as plt
# from main import create_model, NUM_CLASSES, IMG_SIZE, NUM_EPOCHS, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, prepare_image_for_prediction, get_display_string, draw_prediction, label_dict_l, predict, train_model, get_label_dict, train_generator, validation_generator
from main import prepare_image_for_prediction, get_display_string, draw_prediction
# Load the trained model
trained_model_l = tf.keras.models.load_model("trained_model_l.h5")

# Load the label dictionary
with open("label_dict_l.json", "r") as f:
    temp = json.load(f)
    python_dict = json.loads(temp)

video_capture = cv2.VideoCapture(0)
st_image = st.empty()

IMG_SIZE = 255


def real_time_detection(video_capture, st_image):
    while st.checkbox("Enable Real-time Detection"):
        ret_val, frame = video_capture.read()
        print("casse1")

        resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        print("casse2")

        frame_for_pred = prepare_image_for_prediction(resized_frame)
        print("3")
        # Use the loaded model for prediction
        pred_vec = trained_model_l.predict(frame_for_pred)
        print("4")
        pred_class = []
        print("5")
        confidence = np.round(pred_vec.max(), 2)
        print("6")
        if confidence > 0.4:
            pc = pred_vec.argmax()
            pred_class.append((pc, confidence))

            print("7")
        else:
            pred_class.append((0, 0))
            print("8")
        if pred_class:
            txt = get_display_string(pred_class, python_dict)
            frame = draw_prediction(frame, txt)
            print("9")

        st_image.image(frame, channels="BGR", use_column_width=True)
        print("10")


real_time_detection(video_capture, st_image)
