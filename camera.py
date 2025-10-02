import cv2
from models import DrunkClassifier
import os
import time
from db import save_log
import face_recognition
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
MODEL = os.environ.get('MODEL_PATH')

class VideoCamera:
    def __init__(self, location_callback=None):
        # macOS Continuity Camera 경고 해결을 위한 설정
        self.video = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        # 카메라 설정 최적화
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        
        self.classifier = DrunkClassifier(model_path=MODEL)
        self.get_location = location_callback

        if not self.video.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다.")

        self.last_prediction_time = 0
        self.prediction_label = "Sober"
        self.face_detected = False

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        now = time.time()
        
        # 프레임 줄여서 얼굴 검출 (속도 개선)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = []
        if int(now) % 5 == 0:  # 5프레임마다 얼굴 검출
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            # 좌표를 원래 크기로 환산
            face_locations = [(top*2, right*2, bottom*2, left*2) for (top, right, bottom, left) in face_locations]

        # 얼굴 탐지 상태 업데이트
        self.face_detected = len(face_locations) > 0
        
        if len(face_locations) > 0 and (now - self.last_prediction_time) > 3:
            top, right, bottom, left = face_locations[0]
            face_img = frame[top:bottom, left:right]

            label = self.classifier.predict(face_img)
            location = self.get_location() if self.get_location else "Unknown"

            # if label == "Drunk":
            folder = "static/logs"
            os.makedirs(folder, exist_ok=True)
            filename = f"{folder}/{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)

            save_log(label=label, location=location, image_path=filename)

            self.last_prediction_time = now
            self.prediction_label = label

        # 얼굴에 사각형 그리기
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # 화면에 상태 메시지 표시
        if self.face_detected:
            # 얼굴이 탐지된 경우 추론 결과 표시
            display_text = self.prediction_label
            text_color = (0, 255, 0) if self.prediction_label == "Sober" else (0, 0, 255)
        else:
            # 얼굴이 탐지되지 않은 경우
            display_text = "No face detected"
            text_color = (255, 255, 0)  # 노란색
        
        cv2.putText(frame, display_text, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def extract_face(self, frame):
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) > 0:
            top, right, bottom, left = face_locations[0]
            face_img = frame[top:bottom, left:right]
            return face_img
        return None

    def generate(self):
        while True:
            frame = self.get_frame()
            if frame is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
