# DrunkGuard 모델 성능 개선 가이드

## 🎯 문제점 분석

### 원인: 얼굴 검출 크기 불일치
- **학습 데이터**: BlazeFace로 검출된 정확한 얼굴 크기 (224x224)
- **기존 시스템**: Haar Cascade + face_recognition으로 검출된 다양한 크기
- **결과**: 모델 성능 저하 및 일관성 없는 분류 결과

## 🚀 해결책

### 1. BlazeFace 기반 실시간 추론 시스템
- **Video prediction.ipynb 방식 적용**
- **학습 데이터와 동일한 얼굴 검출 방식**
- **더 정확한 얼굴 크기 및 위치**

### 2. 기존 시스템 최적화
- **얼굴 크기 비례 마진 적용**
- **학습 데이터와 유사한 전처리**

## ⚙️ 설정 방법

### 환경변수 설정 (.env)
```bash
# BlazeFace 모드 사용 (기본값: false)
USE_BLAZEFACE=true

# 기존 설정들
PORT=5001
DB_URI=sqlite:///instance/drunk_guard.db
MODEL_PATH=model/bestval.pth
```

### 모델 선택
```python
# app.py에서 자동으로 환경변수에 따라 선택
USE_BLAZEFACE = os.environ.get('USE_BLAZEFACE', 'false').lower() == 'true'
camera = VideoCamera(use_blazeface=USE_BLAZEFACE)
```

## 🧪 성능 테스트

### 테스트 스크립트 실행
```bash
python test_models.py
```

### 테스트 옵션
1. **기존 모델만 테스트**: DrunkClassifier 성능 측정
2. **BlazeFace 모델만 테스트**: 새로운 시스템 성능 측정
3. **두 모델 비교**: 성능 차이 분석

### 예상 결과
- **BlazeFace 모델**: 더 정확한 분류, 일관된 결과
- **기존 모델**: 빠른 처리 속도, 다양한 얼굴 크기 대응

## 📊 기술적 개선사항

### BlazeFace 시스템
```python
# 1. 모델 초기화
self.net = EfficientNetAutoAttB4().eval()
self.face_extractor = FaceExtractor(facedet=BlazeFace())

# 2. 얼굴 검출 및 추출
faces_data = self.face_extractor.process_image(img=pil_image)

# 3. 전처리 (학습 데이터와 동일)
face_tensor = self.transf(image=face)['image']

# 4. 추론
score = torch.sigmoid(self.net(face_tensor))
```

### 기존 시스템 개선
```python
# 얼굴 크기 비례 마진 적용
margin_ratio = 0.3  # 얼굴 크기의 30%
margin_x = int(face_width * margin_ratio)
margin_y = int(face_height * margin_ratio)
```

## 🔧 문제 해결

### BlazeFace 초기화 실패 시
- 자동으로 기존 DrunkClassifier로 폴백
- 콘솔에 오류 메시지 출력
- 시스템 정상 작동 유지

### 성능 최적화
- **비동기 처리**: UI 블로킹 방지
- **프레임 스킵**: 3프레임마다 얼굴 검출
- **캐싱**: 얼굴 위치 정보 재사용
- **큐 관리**: 메모리 사용량 제한

## 📈 성능 지표

### 측정 항목
- **FPS**: 초당 프레임 처리 수
- **얼굴 검출률**: 얼굴이 검출된 프레임 비율
- **추론 실행률**: AI 추론이 실행된 프레임 비율
- **분류 정확도**: Drunk/Sober 분류 일관성

### 권장 설정
- **정확도 우선**: `USE_BLAZEFACE=true`
- **속도 우선**: `USE_BLAZEFACE=false`
- **개발/테스트**: 두 모드 모두 테스트 후 선택

## 🎯 사용 권장사항

1. **초기 설정**: BlazeFace 모드로 테스트
2. **성능 확인**: `test_models.py`로 비교 분석
3. **환경에 맞는 선택**: 하드웨어 성능에 따라 결정
4. **정기적 테스트**: 모델 성능 모니터링

## 🔍 추가 개선 방향

1. **모델 재학습**: 실제 음주 데이터로 fine-tuning
2. **앙상블 방법**: 두 모델 결과 조합
3. **실시간 적응**: 얼굴 크기 자동 조정
4. **사용자 피드백**: 분류 결과 정확도 개선
