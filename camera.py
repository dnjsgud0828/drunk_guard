import cv2
from models import DrunkClassifier
import os
import time
from db import save_detected_image

MODEL = os.environ.get('MODEL_PATH')

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)  # 0은 기본 웹캠
        self.classifier = DrunkClassifier(model_path = MODEL) #classifier

        # 정확한 haarcascade 경로 설정
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if not self.video.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다.")
        if self.face_cascade.empty():
            raise IOError(f"Could not load face cascade from {cascade_path}")
        
        self.last_prediction_time = 0
        self.prediction_label = "Sober"  # 기본값

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        now = time.time()
        face = self.extract_face(frame)

        # if face is not None:
        #     label = self.classifier.predict(face)
            
        #     # ✅ 로그 저장 조건
        #     if label == 'Drunk':
        #         from db import save_detected_image  # 위에 import 되어 있지 않다면 추가
        #         save_detected_image(frame, label='Drunk', location='Gate A')

        #     cv2.putText(frame, label, (30, 30),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # else:
        #     cv2.putText(frame, "No face", (30, 30),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        if face is not None and (now - self.last_prediction_time) > 3:
            label = self.classifier.predict(face)
            save_detected_image(frame, label, location="Main Gate")
            self.last_prediction_time = now
            cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # 얼굴 인식 (옵션), haarcascade classifier
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,       # 1.1 ~ 1.3 (클수록 느리지만 정확)
            minNeighbors=4,        # 낮추면 더 많은 얼굴 탐지
            minSize=(60, 60)       # 작은 얼굴 무시
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 프레임을 JPEG로 인코딩
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
    
    def extract_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60)
        )

        # 얼굴이 하나라도 있으면 첫 번째 얼굴만 사용
        for (x, y, w, h) in faces:
            return frame[y:y+h, x:x+w]
        return None

    def generate(self):
        while True:
            frame = self.get_frame()
            if frame is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
