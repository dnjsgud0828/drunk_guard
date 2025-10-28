import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from architectures import fornet
from architectures.fornet import EfficientNetB4ST
import os
from dotenv import load_dotenv
import sys

load_dotenv()

model_path = os.environ.get('MODEL_PATH')

class DrunkClassifier:
    def __init__(self, model_path, threshold=0.5):
        # GPU/CPU 자동 선택
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.model = EfficientNetB4ST() #모델 생성
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)['net'], strict=False)
        self.model.eval()
        self.model.to(self.device)
        
        self.threshold = threshold
        
        # 이미지 전처리 변환기
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """이미지에서 음주 상태 예측"""
        try:
            # 이미지 전처리
            if isinstance(image, str):
                # 파일 경로인 경우
                image = Image.open(image)
            elif hasattr(image, 'shape'):  # OpenCV 이미지 (numpy array)
                # BGR을 RGB로 변환
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            elif hasattr(image, 'mode'):  # PIL Image인 경우
                # 이미 PIL Image이므로 그대로 사용
                pass
            else:
                raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
            
            # 이미지 크기 확인
            if image.size[0] == 0 or image.size[1] == 0:
                return "Sober"
            
            # 텐서로 변환 및 배치 차원 추가
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 모델 추론
            with torch.no_grad():
                output = self.model(image_tensor)
                
                # 출력 크기에 따라 처리
                if output.shape[1] == 1:  # 단일 출력 (DeepFake 점수)
                    score = torch.sigmoid(output).cpu().numpy().flatten()[0]
                    return "Drunk" if score > self.threshold else "Sober"
                else:  # 다중 출력 (Sober/Drunk 확률)
                    sigmoid_output = torch.sigmoid(output).cpu().numpy().flatten()
                    drunk_score = sigmoid_output[1]
                    return "Drunk" if drunk_score > self.threshold else "Sober"
                    
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Sober"
    
    def set_threshold(self, threshold):
        """임계값 동적 조정"""
        self.threshold = threshold
    
    def get_prediction_details(self, image):
        """예측 상세 정보 반환"""
        try:
            # 이미지 전처리
            if isinstance(image, str):
                # 파일 경로인 경우
                image = Image.open(image)
            elif hasattr(image, 'shape'):  # OpenCV 이미지 (numpy array)
                # BGR을 RGB로 변환
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            elif hasattr(image, 'mode'):  # PIL Image인 경우
                # 이미 PIL Image이므로 그대로 사용
                pass
            else:
                raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
            
            # 이미지 크기 확인
            if image.size[0] == 0 or image.size[1] == 0:
                return {
                    'prediction': 'Sober',
                    'confidence': 0.0,
                    'error': 'Empty image'
                }
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                
                raw_output = output.cpu().numpy().flatten()
                sigmoid_output = torch.sigmoid(output).cpu().numpy().flatten()
                
                if output.shape[1] == 1:
                    confidence = float(sigmoid_output[0])
                    prediction = "Drunk" if confidence > self.threshold else "Sober"
                    probabilities = {
                        'sober': 1 - confidence,
                        'drunk': confidence
                    }
                    return {
                        'prediction': prediction,
                        'confidence': confidence,
                        'raw_output': raw_output,
                        'sigmoid_output': sigmoid_output,
                        'probabilities': probabilities,
                        'output_type': 'single'
                    }
                else:
                    drunk_confidence = float(sigmoid_output[1])
                    prediction = "Drunk" if drunk_confidence > self.threshold else "Sober"
                    probabilities = {
                        'sober': float(sigmoid_output[0]),
                        'drunk': drunk_confidence
                    }
                    return {
                        'prediction': prediction,
                        'confidence': drunk_confidence,
                        'raw_output': raw_output,
                        'sigmoid_output': sigmoid_output,
                        'probabilities': probabilities,
                        'output_type': 'multi'
                    }

        except Exception as e:
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'error': str(e)
            }
