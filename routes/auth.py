from flask import Blueprint, request, jsonify, session, redirect, url_for, render_template
import os
import hashlib
import secrets
from functools import wraps
from .db import create_user, get_user_by_id, update_user


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
            nickname=user_id  # 기본 닉네임으로 user_id 사용
        )
        return jsonify({'success': True, 'message': '회원가입이 완료되었습니다.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'회원가입 중 오류가 발생했습니다: {str(e)}'})


@auth_bp.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


