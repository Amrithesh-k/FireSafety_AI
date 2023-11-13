import traceback
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the trained model
trained_model_l = tf.keras.models.load_model("/trained_model_l.h5")

# Load the label dictionary
label_dict = {0: "default", 1: "fire", 2: "smoke"}

IMG_SIZE = 224


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


def prepare_image_for_prediction(img):
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)


def real_time_detection(video_capture, st_image):
    try:
        while True:
            ret_val, frame = video_capture.read()

            # Ensure the frame is successfully captured
            if not ret_val:
                st.warning("Error: Unable to capture frame.")
                break

            resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame_for_pred = prepare_image_for_prediction(resized_frame)

            frame_for_pred = np.squeeze(frame_for_pred, axis=0)

            pred_vec = trained_model_l.predict(
                np.expand_dims(frame_for_pred, axis=0))

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

            st_image.image(frame, channels="BGR", use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    # Assuming video_capture and st_image are defined before calling real_time_detection
    video_capture = cv2.VideoCapture(0)
    st.title("Real-time Detection App")
    st_image = st.empty()
    real_time_detection(video_capture, st_image)
