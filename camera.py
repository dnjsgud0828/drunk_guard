import cv2
from models import DrunkClassifier
import torch
from PIL import Image
import numpy as np
import os
import time

MODEL = os.environ.get('MODEL_PATH')

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0) #0은 기본 웹캠
        self.classifier = DrunkClassifier(model_path=MODEL)
        if not self.video.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다.")

        #정확한 haarcascade 경로 설정
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise IOError(f"Could not load face cascade from {cascade_path}")

        self.last_prediction_time = 0
        # self.prediction_label = "Sober"  # 기본값

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        now = time.time()
        face = self.extract_face(frame)

        # 3초마다만 모델 추론
        if face is not None and (now - self.last_prediction_time) > 3:
            self.prediction_label = self.classifier.predict(face)
            self.last_prediction_time = now

        # 항상 라벨은 표시되도록
        cv2.putText(frame, self.prediction_label, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255) if self.prediction_label == "Drunk" else (0, 255, 0), 2)

        # 얼굴 위치 표시
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # 프레임을 JPEG로 인코딩
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def extract_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        
        # 얼굴이 하나라도 있으면 첫번쨰 얼굴만 사용
        for (x, y, w, h) in faces:
            return frame[y:y+h, x:x+w]
        
        # 얼굴이 없으면 None 반환
        return None

    def generate(self):
        while True:
            frame = self.get_frame()
            if frame is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
