import torch
from torchvision import transforms
from PIL import Image
import cv2
from architectures.fornet import EfficientNetB4ST, EfficientNetAutoAttB4ST
import os
from dotenv import load_dotenv

load_dotenv()

# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'drunk_guard')))

model_path = os.environ.get('MODEL_PATH')

class DrunkClassifier:
    def __init__(self, model_path, threshold=0.5):
        # GPU/CPU 자동 선택
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.model = EfficientNetB4ST() #모델 생성
        # self.model._fc = torch.nn.Linear(self.model._fc.in_features, 2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)['net'], strict=False) #가중치 load
        self.model.to(self.device)  # 모델을 적절한 디바이스로 이동
        self.model.eval()
        
        # 분류 임계값 설정
        self.threshold = threshold
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict(self, face_img):
        img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            
            # 모델 출력 크기 확인
            print(f"모델 출력 크기: {output.shape}")
            
            # 출력 크기에 따라 처리
            if output.shape[1] == 1:
                # 단일 출력 (DeepFake 탐지용)
                raw_output = output.cpu().numpy().flatten()
                print(f"단일 로짓 출력: {raw_output[0]:.4f}")
                
                # Sigmoid 적용
                sigmoid_output = torch.sigmoid(output).cpu().numpy().flatten()
                score = sigmoid_output[0]
                print(f"단일 Sigmoid 출력: {score:.4f}")
                
                # DeepFake 탐지: 0=Real, 1=Fake
                # 음주 탐지로 변환: Fake(1) = Drunk, Real(0) = Sober
                result = 'Drunk' if score > self.threshold else 'Sober'
                print(f"단일 출력 결과: {result} (점수: {score:.4f}, 임계값: {self.threshold})")
                
                return result
            else:
                # 다중 출력 (2클래스)
                raw_output = output.cpu().numpy().flatten()
                print(f"로짓 출력: Sober={raw_output[0]:.4f}, Drunk={raw_output[1]:.4f}")
                
                # Sigmoid 적용 (0-1 범위로 변환)
                sigmoid_output = torch.sigmoid(output).cpu().numpy().flatten()
                print(f"Sigmoid 출력: Sober={sigmoid_output[0]:.4f}, Drunk={sigmoid_output[1]:.4f}")
                
                # Softmax 적용 (확률 분포로 변환)
                probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()
                print(f"Softmax 확률: Sober={probabilities[0]:.4f}, Drunk={probabilities[1]:.4f}")
                
                # 방법 1: Sigmoid 기반 분류 (Scripts 방식과 동일)
                drunk_score = sigmoid_output[1]
                sigmoid_result = 'Drunk' if drunk_score > self.threshold else 'Sober'
                
                # 방법 2: Softmax 기반 분류
                drunk_prob = probabilities[1]
                softmax_threshold = 0.5
                softmax_result = 'Drunk' if drunk_prob > softmax_threshold else 'Sober'
                
                # 방법 3: Argmax (기존 방식)
                _, pred = torch.max(output, 1)
                argmax_result = 'Drunk' if pred.item() == 1 else 'Sober'
                
                print(f"Sigmoid 결과: {sigmoid_result} (점수: {drunk_score:.4f}, 임계값: {self.threshold})")
                print(f"Softmax 결과: {softmax_result} (확률: {drunk_prob:.4f})")
                print(f"Argmax 결과: {argmax_result}")
                
                # 신뢰도 계산
                confidence = abs(drunk_score - self.threshold) * 2
                print(f"신뢰도: {confidence:.4f}")
                
                # Sigmoid 방식 사용 (Scripts와 동일한 방식)
                return sigmoid_result
    
    def set_threshold(self, threshold):
        """임계값 동적 조정"""
        self.threshold = threshold
        print(f"임계값이 {threshold}로 변경되었습니다.")
    
    def get_prediction_details(self, face_img):
        """상세한 예측 정보 반환"""
        img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            
            if output.shape[1] == 1:
                # 단일 출력
                raw_output = output.cpu().numpy().flatten()
                sigmoid_output = torch.sigmoid(output).cpu().numpy().flatten()
                score = sigmoid_output[0]
                
                prediction = 'Drunk' if score > self.threshold else 'Sober'
                confidence = abs(score - self.threshold) * 2
                
                return {
                    'prediction': prediction,
                    'drunk_score': score,
                    'threshold': self.threshold,
                    'confidence': confidence,
                    'raw_output': raw_output,
                    'sigmoid_output': sigmoid_output,
                    'probabilities': sigmoid_output,  # 단일 출력에서는 sigmoid가 확률 역할
                    'output_type': 'single'
                }
            else:
                # 다중 출력
                raw_output = output.cpu().numpy().flatten()
                sigmoid_output = torch.sigmoid(output).cpu().numpy().flatten()
                probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()
                
                drunk_score = sigmoid_output[1]
                prediction = 'Drunk' if drunk_score > self.threshold else 'Sober'
                confidence = abs(drunk_score - self.threshold) * 2
                
                return {
                    'prediction': prediction,
                    'drunk_score': drunk_score,
                    'threshold': self.threshold,
                    'confidence': confidence,
                    'raw_output': raw_output,
                    'sigmoid_output': sigmoid_output,
                    'probabilities': probabilities,
                    'output_type': 'multi'
                }
