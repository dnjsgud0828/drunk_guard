"""
로그인 관련 기능을 관리하는 모듈
- 일반 로그인 처리
- 카카오톡 로그인 처리
- 세션 관리
"""

import requests
import json
from flask import session, request, jsonify
from functools import wraps
import hashlib
import secrets

# 카카오 API 설정
KAKAO_REST_API_KEY = "YOUR_KAKAO_REST_API_KEY"  # 실제 키로 교체 필요
KAKAO_REDIRECT_URI = "http://localhost:5000/kakao/callback"  # 실제 도메인으로 교체 필요

class LoginManager:
    def __init__(self, db=None):
        self.db = db
        self.users = {}  # 실제로는 데이터베이스에서 관리
        
    def hash_password(self, password):
        """비밀번호 해시화"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', 
                                          password.encode('utf-8'), 
                                          salt.encode('utf-8'), 
                                          100000)
        return salt + password_hash.hex()
    
    def verify_password(self, password, stored_hash):
        """비밀번호 검증"""
        salt = stored_hash[:32]
        stored_password = stored_hash[32:]
        password_hash = hashlib.pbkdf2_hmac('sha256',
                                          password.encode('utf-8'),
                                          salt.encode('utf-8'),
                                          100000)
        return password_hash.hex() == stored_password
    
    def register_user(self, user_id, password, email=None):
        """사용자 회원가입"""
        if user_id in self.users:
            return False, "이미 존재하는 아이디입니다."
        
        password_hash = self.hash_password(password)
        self.users[user_id] = {
            'password': password_hash,
            'email': email,
            'created_at': None  # 실제로는 datetime 사용
        }
        return True, "회원가입이 완료되었습니다."
    
    def login_user(self, user_id, password):
        """일반 로그인 처리"""
        if user_id not in self.users:
            return False, "존재하지 않는 아이디입니다."
        
        stored_hash = self.users[user_id]['password']
        if not self.verify_password(password, stored_hash):
            return False, "비밀번호가 일치하지 않습니다."
        
        # 세션에 사용자 정보 저장
        session['user_id'] = user_id
        session['login_type'] = 'normal'
        session['logged_in'] = True
        
        return True, "로그인 성공"
    
    def get_kakao_auth_url(self):
        """카카오 로그인 URL 생성"""
        kakao_auth_url = (
            f"https://kauth.kakao.com/oauth/authorize?"
            f"client_id={KAKAO_REST_API_KEY}&"
            f"redirect_uri={KAKAO_REDIRECT_URI}&"
            f"response_type=code"
        )
        return kakao_auth_url
    
    def get_kakao_access_token(self, code):
        """카카오 액세스 토큰 획득"""
        token_url = "https://kauth.kakao.com/oauth/token"
        data = {
            'grant_type': 'authorization_code',
            'client_id': KAKAO_REST_API_KEY,
            'redirect_uri': KAKAO_REDIRECT_URI,
            'code': code
        }
        
        response = requests.post(token_url, data=data)
        if response.status_code == 200:
            return response.json().get('access_token')
        return None
    
    def get_kakao_user_info(self, access_token):
        """카카오 사용자 정보 조회"""
        user_info_url = "https://kapi.kakao.com/v2/user/me"
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
        
        response = requests.get(user_info_url, headers=headers)
        if response.status_code == 200:
            return response.json()
        return None
    
    def kakao_login(self, kakao_user_info):
        """카카오 로그인 처리"""
        kakao_id = str(kakao_user_info['id'])
        nickname = kakao_user_info['properties']['nickname']
        profile_image = kakao_user_info['properties'].get('profile_image', '')
        
        # 카카오 ID로 사용자 조회/생성
        user_id = f"kakao_{kakao_id}"
        
        if user_id not in self.users:
            # 새 사용자 등록
            self.users[user_id] = {
                'password': None,  # 카카오 로그인은 비밀번호 없음
                'email': None,
                'kakao_id': kakao_id,
                'nickname': nickname,
                'profile_image': profile_image,
                'login_type': 'kakao'
            }
        
        # 세션에 사용자 정보 저장
        session['user_id'] = user_id
        session['login_type'] = 'kakao'
        session['kakao_id'] = kakao_id
        session['nickname'] = nickname
        session['profile_image'] = profile_image
        session['logged_in'] = True
        
        return True, "카카오 로그인 성공"
    
    def logout_user(self):
        """로그아웃 처리"""
        session.clear()
        return True, "로그아웃 완료"
    
    def is_logged_in(self):
        """로그인 상태 확인"""
        return session.get('logged_in', False)
    
    def get_current_user(self):
        """현재 로그인한 사용자 정보 반환"""
        if not self.is_logged_in():
            return None
        
        user_id = session.get('user_id')
        if user_id and user_id in self.users:
            user_info = self.users[user_id].copy()
            user_info['user_id'] = user_id
            return user_info
        return None

# 데코레이터: 로그인 필요
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        return f(*args, **kwargs)
    return decorated_function

# 전역 로그인 매니저 인스턴스
login_manager = LoginManager()

# Flask 라우트 함수들
def setup_login_routes(app, login_manager_instance=None):
    """Flask 앱에 로그인 라우트 설정"""
    global login_manager
    if login_manager_instance:
        login_manager = login_manager_instance
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            user_id = request.form.get('user_id')
            password = request.form.get('password')
            
            success, message = login_manager.login_user(user_id, password)
            
            if success:
                return jsonify({'success': True, 'message': message})
            else:
                return jsonify({'success': False, 'message': message})
        
        return None  # HTML 템플릿 렌더링은 app.py에서 처리
    
    @app.route('/register', methods=['POST'])
    def register():
        data = request.get_json()
        user_id = data.get('user_id')
        password = data.get('password')
        email = data.get('email')
        
        success, message = login_manager.register_user(user_id, password, email)
        return jsonify({'success': success, 'message': message})
    
    @app.route('/kakao/login')
    def kakao_login_redirect():
        """카카오 로그인 페이지로 리다이렉트"""
        auth_url = login_manager.get_kakao_auth_url()
        return jsonify({'auth_url': auth_url})
    
    @app.route('/kakao/callback')
    def kakao_callback():
        """카카오 로그인 콜백 처리"""
        code = request.args.get('code')
        if not code:
            return jsonify({'success': False, 'message': '인증 코드가 없습니다.'})
        
        # 액세스 토큰 획득
        access_token = login_manager.get_kakao_access_token(code)
        if not access_token:
            return jsonify({'success': False, 'message': '액세스 토큰 획득 실패'})
        
        # 사용자 정보 조회
        user_info = login_manager.get_kakao_user_info(access_token)
        if not user_info:
            return jsonify({'success': False, 'message': '사용자 정보 조회 실패'})
        
        # 로그인 처리
        success, message = login_manager.kakao_login(user_info)
        return jsonify({'success': success, 'message': message})
    
    @app.route('/kakao_login', methods=['POST'])
    def kakao_login_api():
        """카카오 로그인 API (JavaScript에서 호출)"""
        data = request.get_json()
        kakao_id = data.get('kakao_id')
        nickname = data.get('nickname')
        profile_image = data.get('profile_image')
        
        # 카카오 사용자 정보 객체 생성
        kakao_user_info = {
            'id': kakao_id,
            'properties': {
                'nickname': nickname,
                'profile_image': profile_image
            }
        }
        
        success, message = login_manager.kakao_login(kakao_user_info)
        return jsonify({'success': success, 'message': message})
    
    @app.route('/logout', methods=['POST'])
    def logout():
        success, message = login_manager.logout_user()
        return jsonify({'success': success, 'message': message})
    
    @app.route('/user/info')
    @login_required
    def user_info():
        """현재 사용자 정보 반환"""
        user = login_manager.get_current_user()
        return jsonify({'user': user})
    
    @app.route('/check_login')
    def check_login():
        """로그인 상태 확인"""
        is_logged_in = login_manager.is_logged_in()
        user = login_manager.get_current_user() if is_logged_in else None
        return jsonify({
            'logged_in': is_logged_in,
            'user': user
        })

# 유틸리티 함수들
def get_login_manager():
    """로그인 매니저 인스턴스 반환"""
    return login_manager

def init_login_manager(db=None):
    """로그인 매니저 초기화"""
    global login_manager
    login_manager = LoginManager(db)
    return login_manager
