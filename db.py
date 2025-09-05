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

# ----------- CRUD 함수 -----------

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
