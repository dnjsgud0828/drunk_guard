import cv2
from routes.models import DrunkClassifier
import os
import time
from routes.db import save_log
import face_recognition
from datetime import datetime
from dotenv import load_dotenv
import threading
import queue
import numpy as np

load_dotenv()
MODEL = os.environ.get('MODEL_PATH')

class VideoCamera:
    def __init__(self, location_callback=None):
        # macOS Continuity Camera 경고 해결을 위한 설정
        self.video = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        # 카메라 설정 최적화 - 해상도를 적절히 조정
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
        
        # Haar Cascade 분류기 추가 (더 빠른 얼굴 검출)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.classifier = DrunkClassifier(model_path=MODEL)
        self.get_location = location_callback

        if not self.video.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다.")

        # 성능 최적화를 위한 변수들
        self.last_prediction_time = 0
        self.last_face_detection_time = 0
        self.prediction_label = "Sober"
        self.face_detected = False
        self.cached_face_locations = []
        self.frame_count = 0
        
        # 비동기 처리를 위한 큐와 스레드
        self.prediction_queue = queue.Queue(maxsize=2)
        self.prediction_result_queue = queue.Queue(maxsize=2)
        self.prediction_thread = threading.Thread(target=self._prediction_worker, daemon=True)
        self.prediction_thread.start()

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def _prediction_worker(self):
        """비동기 추론 작업을 처리하는 워커 스레드"""
        while True:
            try:
                face_img, location = self.prediction_queue.get(timeout=1)
                if face_img is None:
                    break
                
                # AI 모델 추론 실행
                label = self.classifier.predict(face_img)
                
                # 결과를 큐에 저장
                if not self.prediction_result_queue.full():
                    self.prediction_result_queue.put((label, location))
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Prediction error: {e}")
                continue

    def _detect_faces_hybrid(self, frame):
        """하이브리드 얼굴 검출: Haar Cascade + face_recognition"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1단계: Haar Cascade로 빠른 검출
        haar_faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        face_locations = []
        
        if len(haar_faces) > 0:
            # Haar Cascade로 얼굴이 검출된 경우에만 face_recognition 사용
            # 프레임 크기를 줄여서 처리 속도 향상
            small_frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # face_recognition으로 정확한 위치 검출
            fr_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            
            # 좌표를 원래 크기로 환산
            face_locations = [(int(top*2.5), int(right*2.5), int(bottom*2.5), int(left*2.5)) 
                            for (top, right, bottom, left) in fr_locations]
        
        return face_locations

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        now = time.time()
        self.frame_count += 1
        
        # 얼굴 검출 주기 조정 (3프레임마다)
        face_locations = []
        if self.frame_count % 3 == 0 or (now - self.last_face_detection_time) > 0.5:
            face_locations = self._detect_faces_hybrid(frame)
            self.cached_face_locations = face_locations
            self.last_face_detection_time = now
        else:
            # 캐시된 얼굴 위치 사용
            face_locations = self.cached_face_locations

        # 얼굴 탐지 상태 업데이트
        self.face_detected = len(face_locations) > 0
        
        # 비동기 추론 처리
        if len(face_locations) > 0 and (now - self.last_prediction_time) > 2:
            top, right, bottom, left = face_locations[0]
            
            # 얼굴 영역 확장 (더 나은 추론을 위해)
            margin = 20
            top = max(0, top - margin)
            left = max(0, left - margin)
            bottom = min(frame.shape[0], bottom + margin)
            right = min(frame.shape[1], right + margin)
            
            face_img = frame[top:bottom, left:right]
            
            if face_img.size > 0:
                location = self.get_location() if self.get_location else "Unknown"
                
                # 큐가 가득 차지 않았을 때만 추론 요청
                if not self.prediction_queue.full():
                    self.prediction_queue.put((face_img, location))
                
                self.last_prediction_time = now

        # 추론 결과 확인
        try:
            while not self.prediction_result_queue.empty():
                label, location = self.prediction_result_queue.get_nowait()
                self.prediction_label = label
                
                # 로그 저장
                folder = "static/logs"
                os.makedirs(folder, exist_ok=True)
                filename = f"{folder}/{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                save_log(label=label, location=location, image_path=filename)
                
        except queue.Empty:
            pass

        # 얼굴에 사각형 그리기 (더 두껍고 명확하게)
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
            
            # 얼굴 중심점 표시
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        # 화면에 상태 메시지 표시 (더 큰 폰트와 배경)
        if self.face_detected:
            display_text = self.prediction_label
            text_color = (0, 255, 0) if self.prediction_label == "Sober" else (0, 0, 255)
            confidence_text = f"Faces: {len(face_locations)}"
        else:
            display_text = "No face detected"
            text_color = (0, 255, 255)  # 시안색으로 변경
            confidence_text = "Searching..."
        
        # 텍스트 배경 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        
        # 메인 텍스트
        (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
        cv2.rectangle(frame, (20, 20), (30 + text_width, 30 + text_height + baseline), (0, 0, 0), -1)
        cv2.putText(frame, display_text, (25, 25 + text_height), font, font_scale, text_color, thickness)
        
        # 추가 정보 텍스트
        cv2.putText(frame, confidence_text, (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS 정보 표시
        fps_text = f"FPS: {int(1/(time.time() - getattr(self, 'last_frame_time', now) + 0.001))}"
        cv2.putText(frame, fps_text, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        self.last_frame_time = now

        # JPEG 압축 품질 조정으로 성능 향상
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, jpeg = cv2.imencode('.jpg', frame, encode_param)
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
