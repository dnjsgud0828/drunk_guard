from flask import Blueprint, request, jsonify, session, redirect, url_for, render_template
import os
import hashlib
import secrets
import requests
from functools import wraps
from .db import create_user, get_user_by_id, get_user_by_kakao_id, update_user


auth_bp = Blueprint('auth', __name__)


def login_required(f):
    """로그인이 필요한 라우트를 보호하는 데코레이터"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return salt + password_hash.hex()


def _verify_password(password: str, stored_hash: str) -> bool:
    salt = stored_hash[:32]
    stored_password = stored_hash[32:]
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return password_hash.hex() == stored_password


@auth_bp.route('/login', methods=['POST'])
def login_post():
    user_id = request.form.get('user_id')
    password = request.form.get('password')
    if not user_id or not password:
        return jsonify({'success': False, 'message': '아이디/비밀번호를 입력하세요.'})

    user = get_user_by_id(user_id)
    if not user or not user.password or not _verify_password(password, user.password):
        return jsonify({'success': False, 'message': '아이디 또는 비밀번호가 올바르지 않습니다.'})

    session['user_id'] = user.user_id
    session['login_type'] = user.login_type
    session['nickname'] = user.nickname or user.user_id
    session['logged_in'] = True
    return jsonify({'success': True})


@auth_bp.route('/register', methods=['POST'])
def register_post():
    data = request.get_json(force=True)
    user_id = data.get('user_id')
    password = data.get('password')
    email = data.get('email')

    if not user_id or not password:
        return jsonify({'success': False, 'message': '필수 값이 누락되었습니다.'})
    
    # 중복 아이디 체크
    if get_user_by_id(user_id):
        return jsonify({'success': False, 'message': '이미 존재하는 아이디입니다.'})

    try:
        create_user(
            user_id=user_id,
            password_hash=_hash_password(password),
            email=email,
            login_type='normal'
        )
        return jsonify({'success': True, 'message': '회원가입이 완료되었습니다.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'회원가입 중 오류가 발생했습니다: {str(e)}'})


# Kakao OAuth (서버 사이드 최소 구현)
@auth_bp.route('/kakao/login', methods=['GET'])
def kakao_login_redirect():
    client_id = os.environ.get('KAKAO_REST_API_KEY')
    redirect_uri = os.environ.get('KAKAO_REDIRECT_URI', 'http://localhost:5000/kakao/callback')
    auth_url = (
        'https://kauth.kakao.com/oauth/authorize'
        f'?client_id={client_id}'
        f'&redirect_uri={redirect_uri}'
        '&response_type=code'
    )
    return redirect(auth_url)


@auth_bp.route('/kakao/callback', methods=['GET'])
def kakao_callback():
    code = request.args.get('code')
    if not code:
        return jsonify({'success': False, 'message': '인증 코드가 없습니다.'})

    client_id = os.environ.get('KAKAO_REST_API_KEY')
    redirect_uri = os.environ.get('KAKAO_REDIRECT_URI', 'http://localhost:5000/kakao/callback')

    token_resp = requests.post(
        'https://kauth.kakao.com/oauth/token',
        data={
            'grant_type': 'authorization_code',
            'client_id': client_id,
            'redirect_uri': redirect_uri,
            'code': code,
        },
        timeout=10,
    )
    if token_resp.status_code != 200:
        return jsonify({'success': False, 'message': '토큰 발급 실패'})

    access_token = token_resp.json().get('access_token')
    if not access_token:
        return jsonify({'success': False, 'message': '토큰이 없습니다.'})

    user_resp = requests.get(
        'https://kapi.kakao.com/v2/user/me',
        headers={'Authorization': f'Bearer {access_token}'},
        timeout=10,
    )
    if user_resp.status_code != 200:
        return jsonify({'success': False, 'message': '사용자 정보 조회 실패'})

    info = user_resp.json()
    kakao_id = str(info.get('id'))
    properties = info.get('properties') or {}
    nickname = properties.get('nickname')
    profile_image = properties.get('profile_image')

    # 카카오 ID로 기존 사용자 조회
    user = get_user_by_kakao_id(kakao_id)
    
    if not user:
        # 새 사용자 생성
        user_id = f'kakao_{kakao_id}'
        try:
            user = create_user(
                user_id=user_id,
                password_hash=None,
                email=None,
                nickname=nickname,
                profile_image=profile_image,
                login_type='kakao',
                kakao_id=kakao_id
            )
        except Exception as e:
            return jsonify({'success': False, 'message': f'사용자 생성 중 오류가 발생했습니다: {str(e)}'})
    else:
        # 기존 사용자 정보 업데이트
        update_user(user.user_id, nickname=nickname, profile_image=profile_image)

    session['user_id'] = user.user_id
    session['login_type'] = 'kakao'
    session['nickname'] = user.nickname or user.user_id
    session['logged_in'] = True
    return redirect(url_for('main'))


@auth_bp.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


