# Render 배포 가이드

이 가이드는 DrunkGuard 앱을 Render에 배포하는 방법을 안내합니다.

## 📋 사전 준비사항

1. **GitHub 저장소 준비**
   - 모든 코드가 GitHub에 푸시되어 있어야 합니다
   - `requirements.txt`, `Procfile` 파일이 포함되어 있어야 합니다

2. **Render 계정 생성**
   - https://render.com 에서 계정 생성
   - GitHub 계정으로 연동

## 🚀 배포 단계

### 1단계: PostgreSQL 데이터베이스 생성

1. Render Dashboard 접속
2. **New +** 버튼 클릭
3. **PostgreSQL** 선택
4. 설정:
   - **Name**: `drunk-guard-db` (원하는 이름)
   - **Database**: `drunk_guard`
   - **User**: 자동 생성 또는 지정
   - **Region**: 가장 가까운 지역 선택 (예: Singapore)
   - **PostgreSQL Version**: 최신 버전
   - **Plan**: Free (시작은 무료)
5. **Create Database** 클릭
6. 데이터베이스 생성 완료 후 **Connection String** 복사
   - 형식: `postgresql://user:password@host:port/database`

### 2단계: Web Service 생성

1. Render Dashboard에서 **New +** 버튼 클릭
2. **Web Service** 선택
3. **Connect GitHub** 클릭
4. 저장소 연결 및 선택

### 3단계: Web Service 설정

#### 기본 설정
- **Name**: `drunk-guard` (원하는 이름)
- **Region**: 데이터베이스와 같은 지역 선택
- **Branch**: `main` (또는 기본 브랜치)
- **Root Directory**: 비워두기 (루트에 있으므로)
- **Runtime**: `Python 3`
- **Build Command**: 
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command**: 
  ```bash
  gunicorn app:app --bind 0.0.0.0:$PORT
  ```

#### 환경변수 설정

**Environment** 섹션에서 다음 환경변수들을 추가:

1. **DB_URI**
   - Key: `DB_URI`
   - Value: 1단계에서 복사한 PostgreSQL 연결 문자열
   - 예: `postgresql://user:password@host:port/database`

2. **SECRET_KEY**
   - Key: `SECRET_KEY`
   - Value: 랜덤 문자열 (세션 암호화용)
   - 생성 방법:
     ```python
     import secrets
     print(secrets.token_hex(32))
     ```
   - 또는 온라인 생성기 사용

3. **PORT**
   - Key: `PORT`
   - Value: `10000` (Render가 자동 할당하지만 명시 가능)

4. **MODEL_PATH**
   - Key: `MODEL_PATH`
   - Value: `model/bestval.pth` (기본값)
   - 모델 파일이 다른 경로에 있다면 경로 수정

5. **USE_BLAZEFACE** (선택사항)
   - Key: `USE_BLAZEFACE`
   - Value: `false` (클라우드 환경에서는 사용 안 함)

### 4단계: 고급 설정 (선택사항)

1. **Health Check Path**: `/` (기본값)
2. **Auto-Deploy**: `Yes` (자동 배포 활성화)
3. **Plan**: 
   - **Free**: 무료 (15분 비활성 시 스핀다운)
   - **Starter**: $7/월 (항상 실행)

### 5단계: 배포 시작

1. 모든 설정 완료 후 **Create Web Service** 클릭
2. 배포 과정 확인:
   - 코드 다운로드
   - 의존성 설치 (시간 소요 가능 - PyTorch 등)
   - 서비스 시작

### 6단계: 배포 완료 확인

1. **Logs** 탭에서 배포 로그 확인
2. 성공 메시지 확인:
   ```
   카메라 초기화 실패 (클라우드 환경으로 추정): ...
   카메라 모드: 클라이언트 이미지 전송 모드
   ```
3. **Settings** 탭에서 URL 확인
   - 예: `https://drunk-guard.onrender.com`

## ⚠️ 주의사항

### 1. 모델 파일 크기
- 모델 파일(`model/bestval.pth`)은 GitHub에 있어야 합니다
- 파일이 크면 Git LFS 사용 권장

### 2. 무료 플랜 제한사항
- **스핀다운**: 15분 비활성 시 자동 종료
- 첫 요청 시 **콜드 스타트**: 약 30초~1분 소요
- **메모리**: 512MB 제한

### 3. 데이터베이스 마이그레이션
- 배포 후 자동으로 테이블 생성됨 (`db.create_all()`)
- 기존 데이터 이전이 필요하다면:
  1. 로컬에서 SQLite 데이터 export
  2. PostgreSQL로 import

### 4. 파일 저장 경로
- `static/logs/` 폴더는 컨테이너 재시작 시 삭제될 수 있음
- 영구 저장이 필요하면:
  - Render Disk 사용 (유료 플랜)
  - 또는 클라우드 스토리지 연동 (S3, Cloudflare R2 등)

## 🔧 트러블슈팅

### 배포 실패 시

1. **Logs 확인**
   - 의존성 설치 오류 확인
   - 모델 파일 경로 확인

2. **환경변수 확인**
   - DB_URI 형식 확인
   - 필요한 모든 변수 설정 확인

3. **포트 충돌**
   - PORT 환경변수 제거 (Render가 자동 할당)

### 카메라 접근 문제

- 클라우드 환경에서는 물리적 카메라 접근 불가
- 자동으로 클라이언트 카메라 모드로 전환됨
- 브라우저에서 카메라 권한 허용 필요

### 데이터베이스 연결 실패

1. PostgreSQL 인스턴스가 실행 중인지 확인
2. Connection String 정확성 확인
3. 방화벽 설정 확인 (Render는 자동 처리)

## 📊 모니터링

1. **Dashboard**: 서비스 상태 확인
2. **Logs**: 실시간 로그 확인
3. **Metrics**: CPU, 메모리 사용량 확인

## 🔄 업데이트 배포

1. **자동 배포**: GitHub에 push하면 자동 배포
2. **수동 배포**: Dashboard에서 **Manual Deploy** 클릭

## 💰 비용

### 무료 플랜
- Web Service: 무료 (스핀다운 있음)
- PostgreSQL: 무료 (90일 무료 체험 후 종료)

### 유료 플랜 (권장)
- Starter: $7/월 (항상 실행, 빠른 응답)
- PostgreSQL: $7/월 (영구 저장)

## 📝 체크리스트

배포 전 확인:
- [ ] GitHub에 모든 코드 푸시됨
- [ ] `requirements.txt` 파일 존재
- [ ] `Procfile` 파일 존재
- [ ] PostgreSQL 데이터베이스 생성됨
- [ ] 환경변수 모두 설정됨
- [ ] 모델 파일 경로 확인됨

배포 후 확인:
- [ ] 서비스가 정상 실행됨
- [ ] 데이터베이스 연결 성공
- [ ] 웹사이트 접속 가능
- [ ] 카메라 기능 작동 (클라이언트 모드)
- [ ] 로그인 기능 작동

## 🎉 완료!

배포가 완료되면 Render에서 제공하는 URL로 앱에 접속할 수 있습니다.

---

**추가 도움**: Render 공식 문서 https://render.com/docs

