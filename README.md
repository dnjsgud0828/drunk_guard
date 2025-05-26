# Drunk_Guard
---
취객 구분 프로그램. 동공의 크기 및 혈색 등을 종합적으로 판단하여 취객을 구분

## 프로젝트 구성
본 프로젝트는 "Video Face Manipulation Detection Through Ensemble of CNNs"의 모델을 fine-tuning 하여 판단 모델로 사용.
(<https://github.com/polimi-ispl/icpr2020dfdc>)
EfficientnetB4를 기반으로, Attention mechanism을 접목 후 End-to-End Training과 Siamese Training으로 학습 후 앙상블 기법을 활용해 정확도를 높힘

1. architecture: EfficientnetB4와 Siamese Training에 대한 코드
2. model: 취객 판단 모델. "Dataset of Perceived Intoxicated Faces"의 데이터와 사진데이터를 활용하여 학습을 진행, 학습 방법에 대한 코드는 아래 Repository를 확인
   DIF: <https://sites.google.com/view/difproject/home>,
3. camera.py: 캠에 대한 코드, cv2의 haar cascade 분류기 사용

## 기술 스택
1. Server: Flask
2. Database:
3. OCR: cv2-haar cascade classifier
