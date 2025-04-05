# video_detect.py
import cv2
import os
from mtcnn import MTCNN
from PIL import Image
import numpy as np
import uuid
import tensorflow as tf
tf.compat.v1.enable_eager_execution()


def detect_best_face(video_path):
    detector = MTCNN()
    cap = cv2.VideoCapture(video_path)

    max_area = 0
    best_face = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        for face in faces:
            x, y, w, h = face['box']
            area = w * h
            if area > max_area:
                max_area = area
                best_face = rgb[y:y+h, x:x+w]

    cap.release()

    if best_face is not None:
        filename = "detected.jpg"
        save_path = os.path.join("static/Detected", filename)
        face_img = Image.fromarray(best_face)
        face_img.save(save_path)
        return save_path

    return None
