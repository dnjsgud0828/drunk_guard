# db.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Flask 앱에서 초기화할 DB 객체
db = SQLAlchemy()

# drunk_logs 테이블을 ORM 모델로 정의
class DrunkLog(db.Model):
    __tablename__ = "drunk_logs"

    id = db.Column(db.Integer, primary_key=True)
    label = db.Column(db.String(20), nullable=False)       # Drunk / Sober
    location = db.Column(db.String(100), nullable=True)    # 위치 정보
    image_path = db.Column(db.String(200), nullable=True)  # 이미지 경로
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<DrunkLog {self.label} at {self.timestamp}>"
    
# 유저 테이블
class User(db.Model):
    __tablename__ = "users"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), unique=True, nullable=False)  # 로그인용 아이디
    password = db.Column(db.String(255), nullable=False)  # 해시된 비밀번호
    email = db.Column(db.String(100), nullable=True)  # 이메일 (선택사항)
    nickname = db.Column(db.String(50), nullable=True)  # 닉네임
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<User {self.user_id} at {self.created_at}>"
    
# ----------- Log CRUD 함수 -----------

def save_log(label, location, image_path):
    from app import app   # Flask app 가져오기
    with app.app_context():
        log = DrunkLog(label=label, location=location, image_path=image_path)
        db.session.add(log)
        db.session.commit()

def get_drunk_logs():
    return DrunkLog.query.order_by(DrunkLog.timestamp.desc()).all()

def delete_logs_by_date(date_str):
    db.session.query(DrunkLog).filter(db.func.date(DrunkLog.timestamp) == date_str).delete()
    db.session.commit()

def delete_logs_by_label(label):
    db.session.query(DrunkLog).filter_by(label=label).delete()
    db.session.commit()

def delete_all_logs():
    db.session.query(DrunkLog).delete()
    db.session.commit()

def get_logs_sorted(by="timestamp", order="desc"):
    allowed_fields = {"timestamp": DrunkLog.timestamp, "label": DrunkLog.label}
    col = allowed_fields.get(by)
    if not col:
        raise ValueError("Invalid sort field")
    
    if order.lower() == "asc":
        return DrunkLog.query.order_by(col.asc()).all()
    else:
        return DrunkLog.query.order_by(col.desc()).all()

# ----------- User CRUD 함수 -----------

def create_user(user_id, password_hash, email=None, nickname=None):
    """새 사용자 생성"""
    from app import app
    with app.app_context():
        user = User(
            user_id=user_id,
            password=password_hash,
            email=email,
            nickname=nickname
        )
        db.session.add(user)
        db.session.commit()
        return user

def get_user_by_id(user_id):
    """user_id로 사용자 조회"""
    return User.query.filter_by(user_id=user_id).first()

def update_user(user_id, **kwargs):
    """사용자 정보 업데이트"""
    from app import app
    with app.app_context():
        user = User.query.filter_by(user_id=user_id).first()
        if user:
            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            db.session.commit()
            return user
        return None

def delete_user(user_id):
    """사용자 삭제"""
    from app import app
    with app.app_context():
        user = User.query.filter_by(user_id=user_id).first()
        if user:
            db.session.delete(user)
            db.session.commit()
            return True
        return False

def get_all_users():
    """모든 사용자 조회"""
    return User.query.order_by(User.created_at.desc()).all()
